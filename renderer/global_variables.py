#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
g_render4cnn_root_folder = os.path.dirname(os.path.abspath(__file__))
# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
# Download blender from = https://download.blender.org/release/Blender2.71/
g_blender_executable_path = '/private/home/kalyanv/learning_vision3d/datasets/blender/blender-2.71-linux-glibc211-x86_64/blender' #!! MODIFY if necessary
g_blank_blend_file_path = os.path.join(g_render4cnn_root_folder, 'blank.blend')
g_blender_python_script = os.path.join(g_render4cnn_root_folder, 'render_model_views.py')