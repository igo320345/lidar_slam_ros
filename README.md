# 2D LiDAR Localization

```
<node name="lidar_localization_node" pkg="lidar_slam_ros" type="lidar_localization.py">
        <param name="num_particles" type="int" value="100"/>
        <param name="laser_beams" type="int" value="8"/>
        <param name="laser_sigma_hit" type="double" value="0.2"/>
        <param name="laser_z_hit" type="double" value="0.8"/>
        <param name="laser_z_rand" type="double" value="0.05"/>
        <param name="laser_z_short" type="double" value="0.1"/>
        <param name="laser_z_max" type="double" value="0.05"/>
        <param name="laser_lambda_short" type="double" value="0.1"/>
        <param name="odom_alpha1" type="double" value="0.2"/>
        <param name="odom_alpha2" type="double" value="0.2"/>
        <param name="odom_alpha3" type="double" value="0.2"/>
        <param name="odom_alpha4" type="double" value="0.2"/>
        <param name="base_frame_id" type="string" value="base_link"/>
        <param name="odom_frame_id" type="string" value="odom"/>
        <param name="global_frame_id" type="string" value="map"/>
    </node>
```
