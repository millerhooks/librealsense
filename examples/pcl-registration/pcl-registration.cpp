/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2010, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

/* \author Radu Bogdan Rusu
 * adaptation Raphael Favier*/

#include <boost/make_shared.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>

#include <pcl/io/pcd_io.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>

#include <pcl/features/normal_3d.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>

#include <pcl/visualization/pcl_visualizer.h>

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include "example.hpp" // Include short list of convenience functions for rendering

#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>

#include <string>
#include <map>
#include <algorithm>
#include <mutex>                    // std::mutex, std::lock_guard
#include <cmath>                    // std::ceil

using pcl_ptr = pcl::PointCloud<pcl::PointXYZ>::Ptr;

pcl_ptr points_to_pcl(const rs2::points& points)
{
    pcl_ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    auto sp = points.get_profile().as<rs2::video_stream_profile>();
    cloud->width = sp.width();
    cloud->height = sp.height();
    cloud->is_dense = false;
    cloud->points.resize(points.size());
    auto ptr = points.get_vertices();
    for (auto& p : cloud->points)
    {
        p.x = ptr->x;
        p.y = ptr->y;
        p.z = ptr->z;
        ptr++;
    }

    return cloud;
}


const std::string no_camera_message = "No camera connected, please connect 1 or more";
const std::string platform_camera_name = "Platform Camera";


//convenient typedefs
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;



// Define a new point representation for < x, y, z, curvature >
class MyPointRepresentation : public pcl::PointRepresentation <PointNormalT>
{
    using pcl::PointRepresentation<PointNormalT>::nr_dimensions_;
public:
    MyPointRepresentation ()
    {
        // Define the number of dimensions
        nr_dimensions_ = 4;
    }

    // Override the copyToFloatArray method to define our feature vector
    virtual void copyToFloatArray (const PointNormalT &p, float * out) const
    {
        // < x, y, z, curvature >
        out[0] = p.x;
        out[1] = p.y;
        out[2] = p.z;
        out[3] = p.curvature;
    }
};



class device_container
{
    // Helper struct per pipeline
    struct view_port
    {
        std::map<int, rs2::frame> frames_per_stream;
        rs2::colorizer colorize_frame;
        texture tex;
        rs2::pipeline pipe;
        rs2::pipeline_profile profile;
        rs2::pointcloud pc;
        rs2::points points;
    };

public:


    void enable_device(rs2::device dev)
    {
        std::string serial_number(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
        std::lock_guard<std::mutex> lock(_mutex);

        if (_devices.find(serial_number) != _devices.end())
        {
            return; //already in
        }

        // Ignoring platform cameras (webcams, etc..)
        if (platform_camera_name == dev.get_info(RS2_CAMERA_INFO_NAME))
        {
            return;
        }
        // Create a pipeline from the given device
        rs2::pipeline p;
        rs2::config c;
        c.enable_device(serial_number);
        // Start the pipeline with the configuration
        rs2::pipeline_profile profile = p.start(c);
        // Hold it internally
        _devices.emplace(serial_number, view_port{ {},{},{}, p, profile });

    }

    void remove_devices(const rs2::event_information& info)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        // Go over the list of devices and check if it was disconnected
        auto itr = _devices.begin();
        while(itr != _devices.end())
        {
            if (info.was_removed(itr->second.profile.get_device()))
            {
                itr = _devices.erase(itr);
            }
            else
            {
                ++itr;
            }
        }
    }

    size_t device_count()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        return _devices.size();
    }

    int stream_count()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        int count = 0;
        for (auto&& sn_to_dev : _devices)
        {
            for (auto&& stream : sn_to_dev.second.frames_per_stream)
            {
                if (stream.second)
                {
                    count++;
                }
            }
        }
        return count;
    }

    void poll_frames()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        // Go over all device
        for (auto&& view : _devices)
        {
            // Ask each pipeline if there are new frames available
            rs2::frameset frameset;
            if (view.second.pipe.poll_for_frames(&frameset))
            {
                for (int i = 0; i < frameset.size(); i++)
                {
                    rs2::frame new_frame = frameset[i];
                    int stream_id = new_frame.get_profile().unique_id();
                    view.second.frames_per_stream[stream_id] = view.second.colorize_frame.process(new_frame); //update view port with the new stream
                }
            }
        }
    }
    std::vector<pcl_ptr> get_clouds(int cols, int rows, float view_width, float view_height, window app, bool downsample = true) {
        std::lock_guard<std::mutex> lock(_mutex);
        int stream_no = 0;
        int device_no = 0;
        rs2::points layer_stack[2];
        for (auto &&view : _devices) {
            // For each device get its frames
            PCL_INFO ("Getting Clouds for Device %d\n", device_no);
            device_no++;
            for (auto &&id_to_frame : view.second.frames_per_stream) {
                PCL_INFO ("Getting Stream number %d\n", stream_no);

                // Wait for the next set of frames from the camera
                glfw_state app_state;
                auto frames = view.second.pipe.wait_for_frames();

                auto depth = frames.get_depth_frame();

                // Generate the pointcloud and texture mappings
                view.second.points = view.second.pc.calculate(depth);

                auto color = frames.get_color_frame();
                // For cameras that don't have RGB sensor, we'll map the pointcloud to infrared instead of color
                if (!color)
                    color = frames.get_infrared_frame();

                // Tell pointcloud object to map to this color frame
                view.second.pc.map_to(color);

                // Upload the color frame to OpenGL
                app_state.tex.upload(color);

                layer_stack[stream_no] = view.second.points;
                PCL_INFO ("This Many Points %d\n", view.second.points.size());
                stream_no++;
            }
        }

        glfw_state app_state;
        draw_pointcloud(view_width, view_height, app_state, layer_stack[0]);

        std::vector<pcl_ptr> layers;
        PointCloud::Ptr result(new PointCloud), source, target;
        Eigen::Matrix4f GlobalTransform = Eigen::Matrix4f::Identity(), pairTransform;

        source = points_to_pcl(layer_stack[0]);
        target = points_to_pcl(layer_stack[1]);

        PointCloud::Ptr temp(new PointCloud);

        // Downsample for consistency and speed
        // \note enable this for large datasets
        PointCloud::Ptr src(new PointCloud);
        PointCloud::Ptr tgt(new PointCloud);
        pcl::VoxelGrid<PointT> grid;
        if (downsample) {
            grid.setLeafSize(30.0, 30.0, 30.0);
            grid.setInputCloud(source);
            grid.filter(*src);

            grid.setInputCloud(target);
            grid.filter(*tgt);
        } else {
            src = source;
            tgt = target;
        }


        // Compute surface normals and curvature
        PointCloudWithNormals::Ptr points_with_normals_src(new PointCloudWithNormals);
        PointCloudWithNormals::Ptr points_with_normals_tgt(new PointCloudWithNormals);

        pcl::NormalEstimation<PointT, PointNormalT> norm_est;
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        norm_est.setSearchMethod(tree);
        norm_est.setKSearch(30);

        norm_est.setInputCloud(src);
        norm_est.compute(*points_with_normals_src);
        pcl::copyPointCloud(*src, *points_with_normals_src);

        norm_est.setInputCloud(tgt);
        norm_est.compute(*points_with_normals_tgt);
        pcl::copyPointCloud(*tgt, *points_with_normals_tgt);

        //
        // Instantiate our custom point representation (defined above) ...
        MyPointRepresentation point_representation;
        // ... and weight the 'curvature' dimension so that it is balanced against x, y, and z
        float alpha[4] = {1.0, 1.0, 1.0, 1.0};
        point_representation.setRescaleValues(alpha);

        //
        // Align
        pcl::IterativeClosestPointNonLinear<PointNormalT, PointNormalT> reg;
        reg.setTransformationEpsilon(1e-6);
        // Set the maximum distance between two correspondences (src<->tgt) to 10cm
        // Note: adjust this based on the size of your datasets
        reg.setMaxCorrespondenceDistance(0.1);
        // Set the point representation
        reg.setPointRepresentation(boost::make_shared<const MyPointRepresentation>(point_representation));

        reg.setInputSource(points_with_normals_src);
        reg.setInputTarget(points_with_normals_tgt);



        //
        // Run the same optimization in a loop and visualize the results
        Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity(), prev, targetToSource;
        PointCloudWithNormals::Ptr reg_result = points_with_normals_src;
        reg.setMaximumIterations(20);
        for (int i = 0; i < 256; ++i) {
            PCL_INFO ("Iteration Nr. %d.\n", i);

            // save cloud for visualization purpose
            points_with_normals_src = reg_result;

            // Estimate
            reg.setInputSource(points_with_normals_src);
            reg.align(*reg_result);

            //accumulate transformation between each Iteration
            Ti = reg.getFinalTransformation() * Ti;

            //if the difference between this transformation and the previous one
            //is smaller than the threshold, refine the process by reducing
            //the maximal correspondence distance
            if (fabs((reg.getLastIncrementalTransformation() - prev).sum()) < reg.getTransformationEpsilon())
                reg.setMaxCorrespondenceDistance(reg.getMaxCorrespondenceDistance() - 0.001);

            prev = reg.getLastIncrementalTransformation();

        }

        PCL_INFO ("Made It Out.\n");
        //
        // Get the transformation from target to source
        targetToSource = Ti.inverse();
        PCL_INFO ("Can Invert.\n");
        //
        // Transform target back in source frame
        pcl::transformPointCloud(*target, *temp, targetToSource);
        PCL_INFO ("Can transform.\n");

        //add the source to the transformed target
        *temp += *source;

        pairTransform = targetToSource;

        PCL_INFO ("Made It through the alignment");
        //transform current pair into the global transform
        pcl::transformPointCloud(*temp, *result, GlobalTransform);

        //update the global transform
        GlobalTransform = GlobalTransform * pairTransform;


        layers.push_back(result);
        layers.push_back(result);



    }
        private:
        std::mutex _mutex;
        std::map<std::string, view_port> _devices;
};

int main(int argc, char * argv[]) try {
    // Create a simple OpenGL window for rendering:
    window app(1280, 1024, "CPP PCL Multi-Camera Autoregistration Example");

    device_container connected_devices;

    rs2::context ctx;    // Create librealsense context for managing devices

    // Register callback for tracking which devices are currently connected
    ctx.set_devices_changed_callback([&](rs2::event_information &info) {
        connected_devices.remove_devices(info);
        for (auto &&dev : info.get_new_devices()) {
            connected_devices.enable_device(dev);
        }
    });

    // Initial population of the device list
    for (auto &&dev : ctx.query_devices()) // Query the list of connected RealSense devices
    {
        connected_devices.enable_device(dev);
    }

    while (app) // Application still alive?
    {
        connected_devices.poll_frames();
        auto total_number_of_streams = connected_devices.stream_count();
        if (total_number_of_streams == 0) {
            draw_text(int(std::max(0.f, (app.width() / 2) - no_camera_message.length() * 3)),
                      int(app.height() / 2), no_camera_message.c_str());
            continue;
        }
        if (connected_devices.device_count() == 1) {
            draw_text(0, 10, "Please connect another camera");
        }
        int cols = int(std::ceil(std::sqrt(total_number_of_streams)));
        int rows = int(std::ceil(total_number_of_streams / static_cast<float>(cols)));

        float view_width = (app.width() / cols);
        float view_height = (app.height() / rows);
        PCL_INFO ("%d Devices Detexted.\n", connected_devices.device_count());
            connected_devices.get_clouds(cols, rows, view_width, view_height, app);
    }

    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}



