#!/usr/bin/env python3

import time
import os
import sys
import contextlib
from math import radians
import random
import bpy
from mathutils import Vector

SELECTED_MODEL_ROOT = '/home2/xieyunwei/occupancy_networks/selected_models_pix3d/selected_models_pix3d/'
TEST_RENDERING_PATH = '/home2/xieyunwei/occupancy_networks/selected_models_pix3d/render_test'
RENDERING_MAX_CAMERA_DIST = 1.75
RENDERING_BLENDER_TMP_DIR = '/tmp/blender'
RENDER_RESOLUTION = 500

CAMERA_DIST_RATIO = 0.8

FILE_NAMES = ['gt.off', 'onet.off', 'ours.off']
METHOD_NAMES = [ m.split('.')[0] for m in FILE_NAMES]

def mkdir_p(a):
    if not os.path.exists(a):
        os.mkdir(a) 

class BaseRenderer:
    model_idx   = 0

    def __init__(self):
        # bpy.data.scenes['Scene'].render.engine = 'CYCLES'
        # bpy.context.scene.cycles.device = 'GPU'
        # bpy.context.user_preferences.system.compute_device_type = 'CUDA'
        # bpy.context.user_preferences.system.compute_device = 'CUDA_1'

        # changing these values does affect the render.

        # remove the default cube
        bpy.ops.object.select_pattern(pattern="Cube")
        bpy.ops.object.delete()

        render_context = bpy.context.scene.render
        world  = bpy.context.scene.world
        camera = bpy.data.objects['Camera']
        light_1  = bpy.data.objects['Lamp']
        light_1.data.type = 'HEMI'

        # set the camera postion and orientation so that it is in
        # the front of the object
        camera.location       = (1, 0, 0)
        camera.rotation_mode  = 'ZXY'
        camera.rotation_euler = (0, radians(90), radians(90))

        # parent camera with a empty object at origin
        org_obj                = bpy.data.objects.new("RotCenter", None)
        org_obj.location       = (0, 0, 0)
        org_obj.rotation_euler = (0, 0, 0)
        bpy.context.scene.objects.link(org_obj)

        camera.parent = org_obj  # setup parenting

        # render setting
        render_context.resolution_percentage = 100
        world.horizon_color = (1, 1, 1)  # set background color to be white

        # set file name for storing rendering result
        self.result_fn = '%s/render_result_%d.png' % ('/tmp', os.getpid())
        bpy.context.scene.render.filepath = self.result_fn

        # new settings
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'
        bpy.context.scene.render.image_settings.use_zbuffer = True
        
        self.render_context = render_context
        self.org_obj = org_obj
        self.camera = camera
        self.light = light_1
        self._set_lighting()

    def initialize(self, models_fn, viewport_size_x, viewport_size_y):
        self.models_fn = models_fn
        self.render_context.resolution_x = viewport_size_x
        self.render_context.resolution_y = viewport_size_y

    def _set_lighting(self):
        pass

    def setViewpoint(self, azimuth, altitude, yaw, distance_ratio, fov):
        self.org_obj.rotation_euler = (0, 0, 0)
        self.light.location = (distance_ratio *
                               (RENDERING_MAX_CAMERA_DIST + 2), 0, 0)
        self.camera.location = (distance_ratio *
                                RENDERING_MAX_CAMERA_DIST, 0, 0)
        self.org_obj.rotation_euler = (radians(-yaw),
                                       radians(-altitude),
                                       radians(-azimuth))

    def setTransparency(self, transparency):
        """ transparency is either 'SKY', 'TRANSPARENT'
        If set 'SKY', render background using sky color."""
        self.render_context.alpha_mode = transparency

    def selectModel(self):
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_pattern(pattern="RotCenter")
        bpy.ops.object.select_pattern(pattern="Lamp*")
        bpy.ops.object.select_pattern(pattern="Camera")
        bpy.ops.object.select_all(action='INVERT')

    def printSelection(self):
        print(bpy.context.selected_objects)

    def clearModel(self):
        self.selectModel()
        bpy.ops.object.delete()

        # The meshes still present after delete
        for item in bpy.data.meshes:
            bpy.data.meshes.remove(item)
        for item in bpy.data.materials:
            bpy.data.materials.remove(item)

    def setModelIndex(self, model_idx):
        self.model_idx = model_idx

    def loadModel(self, file_path=None):
        if file_path is None:
            file_path = self.models_fn[self.model_idx]

        if file_path.endswith('obj'):
            bpy.ops.import_scene.obj(filepath=file_path)
        elif file_path.endswith('3ds'):
            bpy.ops.import_scene.autodesk_3ds(filepath=file_path)
        elif file_path.endswith('dae'):
            # Must install OpenCollada. Please read README.md
            bpy.ops.wm.collada_import(filepath=file_path)
        elif file_path.endswith('ply'):
            bpy.ops.import_mesh.ply(filepath=file_path)
        elif file_path.endswith('off'):
            bpy.ops.import_mesh.off(filepath=file_path)
        else:
            raise Exception("Loading failed: %s Model loading for type %s not Implemented" %
                            (file_path, file_path[-4:]))

        if not file_path.endswith('obj'):
            ob = bpy.context.scene.objects.active
            ob.rotation_euler = (radians(90), 0, 0)

    def normalizeObject(self):
        obj = bpy.context.scene.objects.active

        #bpy.ops.object.transform_apply( rotation = True, scale = True )

        Xs = [vertex.co[0] for vertex in obj.data.vertices]
        Ys = [vertex.co[1] for vertex in obj.data.vertices]
        Zs = [vertex.co[2] for vertex in obj.data.vertices]

        minX = min(Xs)
        minY = min(Ys)
        minZ = min(Zs)
        maxX = max(Xs)
        maxY = max(Ys)
        maxZ = max(Zs)

        vCenter = Vector([(minX + maxX) / 2., (minY + maxY) / 2., (minZ + maxZ) / 2.])
        maxLength = max([maxX - minX, maxY - minY, maxZ - minZ])

        for v in obj.data.vertices:
            v.co -= vCenter 
            v.co /= maxLength

    def render(self, load_model=True, clear_model=True, resize_ratio=None,
               return_image=True, image_path=os.path.join(RENDERING_BLENDER_TMP_DIR, 'tmp.png'), normalize=False):
        """ Render the object """
        if load_model:
            self.loadModel()

        # resize object
        self.selectModel()
        if resize_ratio:
            bpy.ops.transform.resize(value=resize_ratio)
        if normalize:
            self.normalizeObject()

        self.result_fn = image_path
        bpy.context.scene.render.filepath = image_path
        bpy.ops.render.render(write_still=True)  # save straight to file

        if resize_ratio:
            bpy.ops.transform.resize(value=(1/resize_ratio[0],
                1/resize_ratio[1], 1/resize_ratio[2]))

        if clear_model:
            self.clearModel()


class ShapeNetRenderer(BaseRenderer):

    def __init__(self):
        super().__init__()
        self.setTransparency('TRANSPARENT')

    def _set_lighting(self):
        # Create new lamp datablock
        light_data = bpy.data.lamps.new(name="New Lamp", type='HEMI')

        # Create new object with our lamp datablock
        light_2 = bpy.data.objects.new(name="New Lamp", object_data=light_data)
        bpy.context.scene.objects.link(light_2)

        # put the light behind the camera. Reduce specular lighting
        self.light.location       = (0, -2, 2)
        self.light.rotation_mode  = 'ZXY'
        self.light.rotation_euler = (radians(45), 0, radians(90))
        self.light.data.energy = 0.7

        light_2.location       = (0, 2, 2)
        light_2.rotation_mode  = 'ZXY'
        light_2.rotation_euler = (-radians(45), 0, radians(90))
        light_2.data.energy = 0.7



import argparse
def main():
    file_paths = []
    file_roots = []
    
    class_ids = os.listdir(SELECTED_MODEL_ROOT)
    for class_id in class_ids:
        class_root = os.path.join(SELECTED_MODEL_ROOT, class_id)
        if not os.path.isdir(class_root):
            continue

        modelnames = os.listdir(class_root)
        for modelname in modelnames:
            model_root = os.path.join(class_root, modelname)

            test_obj = os.path.join(model_root, FILE_NAMES[0])
            if not os.path.exists(test_obj):
                continue

            file_roots.append(model_root)

            for file_name in FILE_NAMES:
                cur_obj = os.path.join(model_root, file_name)
                file_paths.append(cur_obj)

    renderer = ShapeNetRenderer()
    renderer.initialize(file_paths, RENDER_RESOLUTION, RENDER_RESOLUTION)

    single_model_render_count = len(FILE_NAMES)
    for i, curr_model_root in enumerate(file_roots):
        image_path = os.path.join(curr_model_root, 'renderings')
        mkdir_p(image_path)

        for k in range(single_model_render_count):
            renderer.setModelIndex(i * single_model_render_count + k)
            renderer.setViewpoint(30, 30, 0, CAMERA_DIST_RATIO, 25)

            cur_image_path = os.path.join(image_path, '%s.png' % METHOD_NAMES[k])

            renderer.render(load_model=True, return_image=False,
                    clear_model=True, image_path=cur_image_path, normalize=(METHOD_NAMES[k] == 'gt'))

            print('Saved at %s' % cur_image_path)

        print('finished %d/%d' % (i, len(file_roots)))

def test():
    """Test function"""
    # Modify the following file to visualize the model
    #dn = '/home2/xieyunwei/occupancy_networks/external/ShapeNetCore.v1/02958343/'
    #model_id = ['2c981b96364b1baa21a66e8dfcce514a']
    mkdir_p(TEST_RENDERING_PATH)

    class_id = 'chair'
    model_id = ['0358']
    file_paths = []
    for m_id in model_id:
        for object_name in FILE_NAMES:
            file_paths.append(os.path.join(SELECTED_MODEL_ROOT, class_id, m_id, object_name))

    renderer = ShapeNetRenderer()
    renderer.initialize(file_paths, RENDER_RESOLUTION, RENDER_RESOLUTION)

    single_model_render_count = len(FILE_NAMES)
    for i, curr_model_id in enumerate(model_id):
        start_time = time.time()
        image_path = os.path.join(TEST_RENDERING_PATH, '%s_%s' % (class_id, curr_model_id))
        mkdir_p(image_path)

        for k in range(single_model_render_count):
            renderer.setModelIndex(i * single_model_render_count + k)
            renderer.setViewpoint(30, 30, 0, CAMERA_DIST_RATIO, 25)

            cur_image_path = os.path.join(image_path, '%s.png' % METHOD_NAMES[k])

            renderer.render(load_model=True, return_image=False,
                    clear_model=True, image_path=cur_image_path, normalize=(METHOD_NAMES[k] == 'gt'))

            print('Saved at %s' % cur_image_path)

        end_time = time.time()
        print('Cost ', end_time - start_time, ' secs')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Render according to a task_split_file')
    parser.add_argument('--test', action='store_true', help='test')

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    if args.test:
        test()
    else:
        main()
