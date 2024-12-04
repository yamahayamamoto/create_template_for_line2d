
import os
import sys

"""
if sys.platform.startswith('win32'):
    # Windowsの場合の処理
    print("Windows")
elif sys.platform.startswith('linux'):
    # Linuxの場合の処理
    print("Linux")
    # GPU 加速レンダリングを事項するためにEGLを使用する
    # display managerなしで加速レンダリングが可能となる
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    # もしくは以下のコマンドでプログラムを実行する
    # PYOPENGL_PLATFORM=egl python render_pyrender.py
    print(os.environ['PYOPENGL_PLATFORM'])
else:
    # その他のプラットフォームの場合の処理
    print("Other")
"""
import pyrender
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import cv2
import time
from tqdm import tqdm
from utils import inout
import copy
from PIL import Image
import json
from werkzeug.utils import cached_property


from pathlib import Path


"""
## カメラの姿勢からpyrenderでレンダリングする
python pyrender\rendering_images_by_pyrender.py --path_camera_intrinsic_param pyrender\dataset\mydata\camera.json
--path_camera_coordinates pyrender\result\camera_coordinates\95D95-06010.npy --mesh pyrender\dataset\mydata\models\95D95-06010\95D95-06010.ply --output_dir pyrender\result
"""
def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--mesh", type=str, default=None, help='object ID to render')
    parser.add_argument("--path_camera_coordinates", type=str, default='./output/mydata',
                        help='path to camera intrinsic param')
    parser.add_argument("--path_camera_intrinsic_param", type=str, default='./dataset/mydata',
                        help='path to camera 6D poses')
    parser.add_argument("--output_dir", type=str, default='./result/train_symmetry/',
                        help='output data directory')
    parser.add_argument("--result_dirname", type=str, default='train',
                        help='output data directory name')

    # -- Data options
    parser.add_argument("--axial_angle_division_number", type=int, default=18,
                        help='division number of axial angle[0,2*pi]')
    parser.add_argument("--background_color", nargs=4, type=float, default=[0.0,0.0,0.0,0.0],
                    help='rendering background color')
    return parser



class RenderedImageInputPath:
    def __init__(self, path_camera_intrinsic_param, path_camera_coordinates, path_model):
        self._path_camera_intrinsic_param = path_camera_intrinsic_param
        self._path_camera_coordinates = path_camera_coordinates
        self._path_model = path_model

    @property
    def path_cam_params(self):
        return self._path_camera_intrinsic_param

    @property
    def path_camera_coordinates(self):
        return self._path_camera_coordinates

    @property
    def path_model(self):
        return self._path_model

    @cached_property
    def obj_id(self):
        target_obj_filename = os.path.basename(self._path_model)
        obj_id = os.path.splitext(target_obj_filename)[0]
        return obj_id

    @property
    def path_camera_intrinsic_param(self):
        return self._path_camera_intrinsic_param


class RenderedImageOutputPath:
    def __init__(self, output_dir, result_dirname, obj_id):
        self.output_dir = Path(output_dir)
        self.result_dirname = result_dirname
        self.obj_id = obj_id
        self.out_path = self.output_dir / self.result_dirname

        # フォルダが存在しない場合は作成
        if not self.out_path.exists():
            self.out_path.mkdir(parents=True)

        # self._out_rgb_dir = os.path.join(self.out_path , f'{self.obj_id:06d}' , 'rgb')
        self._out_rgb_dir = os.path.join(self.out_path , f'{self.obj_id}' , 'rgb')

        # フォルダが存在しない場合は作成
        if not os.path.exists(self._out_rgb_dir):
            os.makedirs(self._out_rgb_dir)

        # self._out_depth_dir = os.path.join(self.out_path , f'{self.obj_id:06d}' , 'depth')
        self._out_depth_dir = os.path.join(self.out_path , f'{self.obj_id}' , 'depth')

        # フォルダが存在しない場合は作成
        if not os.path.exists(self._out_depth_dir):
            os.makedirs(self._out_depth_dir)

        # self._out_mask_dir = os.path.join(self.out_path , f'{self.obj_id:06d}' , 'mask')
        self._out_mask_dir = os.path.join(self.out_path , f'{self.obj_id}' , 'mask')


        # フォルダが存在しない場合は作成
        if not os.path.exists(self._out_mask_dir):
            os.makedirs(self._out_mask_dir)


    @property
    def out_rgb_tpath(self):
        return os.path.join(self._out_rgb_dir, '{im_id:06d}.png')

    @property
    def out_depth_tpath(self):
        return os.path.join(self._out_depth_dir, '{im_id:06d}.png')

    @property
    def out_mask_tpath(self):
        return os.path.join(self._out_mask_dir, '{im_id:06d}.png')

    @property
    def out_scene_camera_tpath(self):
        # return os.path.join(self.out_path, f'{self.obj_id:06d}', 'scene_camera.json')
        return os.path.join(self.out_path, f'{self.obj_id}', 'scene_camera.json')

    @property
    def out_scene_gt_tpath(self):
        # return os.path.join(self.out_path, f'{self.obj_id:06d}', 'scene_gt.json')
        return os.path.join(self.out_path, f'{self.obj_id}', 'scene_gt.json')

    @property
    def out_profile_tpath(self):
        return os.path.join(self.out_path, 'profile.json')


def overlay_depth_edges(rgb_image, depth_image):
    from PIL import ImageEnhance,Image
    enhancer = ImageEnhance.Contrast(Image.fromarray(rgb_image))
    return np.array(enhancer.enhance(1.5))
    # depth_image=copy.copy(depth_image)
    # depth_image[depth_image == 0] = 0

    # depth_image = np.where(depth_image!=0,np.max(depth_image) - depth_image,0)
    # depth_image_uint8 = np.array(255 * (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image)), dtype=np.uint8)
    # depth_image_uint8 = cv2.medianBlur(depth_image_uint8, 7)
    # def sToneCurve(img):
    #     x = np.arange(256)
    #     # S字カーブを適当に定義
    #     y = ((np.sin(np.pi * (x/255 - 0.5)) + 1)/2 * 255).astype('int')
    #     return np.array(cv2.LUT(img, y), dtype=np.uint8)
        
    # depth_image_high_enhance = sToneCurve(depth_image_uint8)
    # depth_image_high_enhance[np.median(depth_image_high_enhance[depth_image_high_enhance != 0])>depth_image_high_enhance]=0
    
    # plt.imshow(depth_image_high_enhance)
    # plt.title("depth_image_high_enhance")
    # plt.show()
    
    # tmp = sToneCurve(-depth_image_uint8)
    # depth_image_low_enhance=copy.copy(depth_image_uint8)
    # depth_image_low_enhance[depth_image_high_enhance!=0]=0
    
    # depth_image_low_enhance[np.median(depth_image_low_enhance[depth_image_low_enhance != 0])>depth_image_low_enhance]=0
    # #depth_image_uint8 = np.array(255 * (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image)), dtype=np.uint8)
    
    # plt.imshow(depth_image_low_enhance)
    # plt.title("depth_image_low_enhance")
    # plt.show()
    # depth_image_low_enhance[depth_image_uint8!=0] = 255-depth_image_low_enhance[depth_image_uint8!=0]
    # plt.imshow(depth_image_low_enhance)
    # plt.title("depth_image_low_enhance inv")
    # plt.show()
    
    # plt.imshow(depth_image_high_enhance+depth_image_low_enhance)
    # plt.title("depth_image_low_enhance")
    # plt.show()
    

    # result_image = cv2.addWeighted(rgb_image, 0.7, cv2.cvtColor(depth_image_high_enhance, cv2.COLOR_GRAY2BGR), 0.3, 0)
    # result_image = cv2.addWeighted(result_image, 0.8, cv2.cvtColor(depth_image_low_enhance, cv2.COLOR_GRAY2BGR), 0.2, 0)
    # # 結果をクリップ（0から255の範囲に制限）
    # result_image = np.clip(result_image, 0, 255).astype(np.uint8)
    
    # #plt.imshow(result_image)
    # #plt.title("result_image")
    # #plt.show()
    
    # return result_image

def overlay_depth_edges2(rgb_image, depth_image):
    """
    オーバーレイされたRGB画像を生成する関数。

    Args:
        rgb_image (np.ndarray): RGB画像 (BGRフォーマット)。
        depth_image (np.ndarray): 深度画像 (グレースケール)。

    Returns:
        np.ndarray: オーバーレイされたRGB画像。

    Usage:
        # 画像の読み込み
        rgb_image = cv2.imread('path_to_rgb_image.png')
        depth_image = cv2.imread('path_to_depth_image.png', cv2.IMREAD_UNCHANGED)
        result_image = overlay_depth_edges(rgb_image, depth_image)
        cv2.imshow('Overlay Image', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    """
    def sToneCurve(img):
        x = np.arange(256)
        # S字カーブを適当に定義
        y = ((np.sin(np.pi * (x/255 - 0.5)) + 1)/2 * 255).astype('int')
        return np.array(cv2.LUT(img, y), dtype=np.uint8)
    depth_image=copy.copy(depth_image)
    #med_val = np.min(depth_image[depth_image != 0])+40
    med_val =0#np.median(depth_image[depth_image != 0])
    depth_image[depth_image == 0] = med_val

    depth_image = np.where(depth_image!=med_val,np.max(depth_image) - depth_image,0)
    # 深度画像の前処理
    depth_image = np.array(255 * (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image)), dtype=np.uint8)
    def polygonalToneWithIni(input_img,k,a):
        return a + np.dot(k,input_img  ).astype(np.uint8)
    def gammaTone(input_img,gamma):
        output_float = 255 * np.power(input_img / 255, gamma) # 計算結果をいったん実数型(float)で保持
        return output_float.astype(np.uint8).astype(np.uint8)
    #depth_image = sToneCurve(depth_image)
    #depth_image = polygonalToneWithIni(depth_image,0.5,125)
    depth_image = gammaTone(depth_image,0.3)
    
    
    
    #depth_image = cv2.bilateralFilter(depth_image, 11, 12, 12)
    # depth_image = cv2.bilateralFilter(depth_image, 11, 12, 12)

    depth_image = cv2.medianBlur(depth_image, 7)
    
    # ソベルフィルタを使用してエッジを計算
    sobel_x = cv2.Sobel(depth_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(depth_image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    edges = cv2.convertScaleAbs(magnitude)
    
    #　細線化処理
    # edges   =   cv2.ximgproc.thinning(edges, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    # plt.imshow(depth_image)
    # plt.title("Depth Image")
    # plt.show()

    # plt.imshow(edges)
    # plt.title("Edges")
    # plt.show()
    # 大津の２値化
    # ret, binary_edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # kernel = np.ones((3, 3), np.uint8)
    # binary_edges = cv2.dilate(binary_edges, kernel, iterations=1)
    # エッジをRGB画像にオーバーレイ
    #result_image = cv2.addWeighted(rgb_image, 0.4, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.6, 0)
    #result_image = cv2.addWeighted(rgb_image, 0.6, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.4, 0)
    result_image = cv2.addWeighted(rgb_image, 0.7, cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR), 0.3, 0)
    # 結果をクリップ（0から255の範囲に制限）
    result_image = np.clip(result_image, 0, 255).astype(np.uint8)
    
    
    return result_image


class Render:

    def __init__(self,
                 input_paths,
                 output_paths,
                 axial_angle_division_number=18,
                 bg_color=(0.0, 0.0, 0.0, 1.0)):

        self._output_paths = output_paths
        self._bg_color       = bg_color
        self._obj_id = input_paths.obj_id
        self._objHcams = self._load_posedataset(input_paths.path_camera_coordinates)
        self.rgb_renderer    = RenderPyrender(input_paths.path_model, input_paths.path_camera_intrinsic_param, "rgb", self._bg_color)
        self.depth_renderer  = RenderPyrender(input_paths.path_model, input_paths.path_camera_intrinsic_param, "depth",  self._bg_color)
        self._axial_angles  = np.linspace(0, 2.0*np.pi, axial_angle_division_number)

    def _load_posedataset(self, path_camera_coordinates):

        objHcams = np.load(path_camera_coordinates)
        return objHcams

    def render(self): 

        print("render object image ...") 

        self.rgb_renderer.remove_model_from_scene() 
        self.rgb_renderer.set_model_on_scene(self.rgb_renderer.model_mesh) 
        self.depth_renderer.remove_model_from_scene() 
        self.depth_renderer.set_model_on_scene(self.depth_renderer.model_mesh) 
  
        scene_camera = {} 
        scene_gt = {} 

        im_id=0 

        for objHcam  in tqdm(self._objHcams): 

            self.rgb_renderer.change_camera_pose(objHcam) 
            self.depth_renderer.change_camera_pose(objHcam) 

            rgb, _, _ = self.rgb_renderer._render_imgs() 
            _, depth, mask = self.depth_renderer._render_imgs() 

            # Super-sampling anti-aliasing (SSAA)のために拡大していた画像をもとのサイズにリサイズする 
            rgb = cv2.resize(rgb, self.depth_renderer.dp_camera['im_size'], interpolation=cv2.INTER_AREA) 
            #rgb = overlay_depth_edges(rgb, depth) 
            #rgb = cv2.medianBlur(rgb,7)
            #rgb = overlay_depth_edges(rgb, depth) 
            mask  = (depth > 0).astype(np.uint8) * 255 

            for axial_angle in self._axial_angles: 

                direction = copy.deepcopy(objHcam[:3,2]) # z-axis 
                rotation_vector = axial_angle * direction 

                # ロドリゲスの回転公式で回転ベクトルを回転行列に変換 
                R_inplane, _ = cv2.Rodrigues(rotation_vector) 
                H_inplane = np.eye(4) 
                H_inplane[:3, :3] = R_inplane 
                objHcam_plus_inplane = np.linalg.inv(H_inplane) @ copy.deepcopy(objHcam) 


                # realsenseの右x,下y、奥zの座標に合わせ（あとで画像と一緒に保存するため） 
                objHcam_aligned_for_realsense =  self.align_camera_coordinate_for_realsense(objHcam_plus_inplane) 

                rotate_rgb = self._rotate2d_img(rgb, axial_angle, self._bg_color[:3], is_rgb=True) 
                rotate_depth = self._rotate2d_img(depth, axial_angle, np.array([0,0,0]), is_rgb=False) 
                rotate_mask  = (rotate_depth > 0).astype(np.uint8) * 255 


                out_rgb_path = self._output_paths.out_rgb_tpath.format(im_id=im_id) 
                inout.save_im(out_rgb_path, rotate_rgb) 
                #cv2.imwrite(out_rgb_path, bgr) 


                out_depth_path = self._output_paths.out_depth_tpath.format(im_id=im_id) 
                inout.save_depth(out_depth_path, rotate_depth) 

                out_mask_path = self._output_paths.out_mask_tpath.format(im_id=im_id) 
                # inout.save_im(out_mask_path, mask) 
                cv2.imwrite(out_mask_path, rotate_mask) 

                distance_surface2cog = self.rgb_renderer.compute_distance_difference(objHcam_plus_inplane, depth, mask)

                # rgb_renderのカメラ内部パラメータはSSAAで変更したのでdeth_renderを使用する 
                scene_camera[im_id] = {'cam_K': self.depth_renderer.dp_camera['K'], 
                                        'depth_scale': self.depth_renderer.dp_camera['depth_scale']} 

                scene_gt[im_id] = [{'cam_R_w2c': objHcam_aligned_for_realsense[:3,:3].tolist(), 
                                    'cam_t_w2c': objHcam_aligned_for_realsense[:3,3].tolist(), 
                                    'obj_id'   : self._output_paths.obj_id,
                                    'distance_surface2cog':distance_surface2cog}] 
                
                im_id +=1 

        # Save metadata. 
        inout.save_scene_camera(self._output_paths.out_scene_camera_tpath, scene_camera) 
        inout.save_scene_gt(self._output_paths.out_scene_gt_tpath, scene_gt) 
        print("the number of image:",len(scene_camera)) 

    def align_camera_coordinate_for_realsense(self, matrix): 

        R_x_180 = np.array([[1,  0,  0, 0], 
                            [0, -1,  0, 0], 
                            [0,  0, -1, 0], 
                            [0,  0,  0, 1]]) 
        matrix_update =  matrix @ R_x_180 

        return matrix_update 


    def _rotate2d_img(self, img, rotate_angle_rad, fill_color, is_rgb=False):
        """画像を2Dで回転させる

        is_rgb rgb画像の場合はTrueをdepthの場合はFalseを
        """
        h,w = img.shape[:2]
        # 回転中心
        rotation_center = (w//2, h//2)

        # 回転後の未定義領域の色
        fill_color = [int(255*rgb) for rgb in fill_color]

        rotation_matrix = cv2.getRotationMatrix2D(rotation_center, np.rad2deg(rotate_angle_rad), 1)

        if(is_rgb):
            # RGB画像ではバイリニア補間のほうが最近傍補間よりノイズが少ない
            rotate2d_img = cv2.warpAffine(img, rotation_matrix, (w, h), borderValue=fill_color, flags=cv2.INTER_LINEAR)
        else:
            # 深度画像では、バイリニア補間より最近傍補間の方がノイズが少ない
            rotate2d_img = cv2.warpAffine(img, rotation_matrix, (w, h), borderValue=fill_color, flags=cv2.INTER_NEAREST)

        return rotate2d_img


    def transform_img_for_matching_realsense(self, img, rotate_angle_rad, fill_color, is_rgb=False):

        yflip = cv2.flip(img, 1)

        return self._rotate2d_img(yflip, rotate_angle_rad, fill_color, is_rgb=is_rgb)


    def _convert_bw_left_and_right_hand(self, H):
        """右手系と左手系を入れ替える(z軸をフリップするだけ)
        """
        H_copy = copy.deepcopy(H)
        H_copy[:,2] = -H_copy[:,2]
        return H_copy


class RenderPyrender:

    def __init__(self, path_model, cam_params_path, mode="rgb", bg_color=[1.0,1.0,1.0,1.0]):

        self.mode = mode
        self._bg_color = np.array(bg_color)
        self.load_model(path_model)

        # すべてのターゲットは原点にある
        self._model_pose = np.eye(4)

        self.dp_camera = {
            # Path to a file with camera parameters.
            'cam_params_path': cam_params_path,
        }

        self.dp_camera.update(inout.load_cam_params(cam_params_path))

        self._set_camera()
        self._set_scene()
        self._set_camera_on_scene()

        self.model_node = None
        self.light_nodes = []

    def _set_camera(self):

        # Super-sampling anti-aliasing (SSAA) - the RGB image is rendered at ssaa_fact
        # times higher resolution and then down-sampled to the required resolution.
        # Ref: https://github.com/vispy/vispy/wiki/Tech.-Antialiasing
        ssaa_fact = 4

        if(self.mode=="rgb"):

            # Image size and K for the RGB image (potentially with SSAA).
            im_size_rgb = [int(round(x * float(ssaa_fact))) for x in self.dp_camera['im_size']]
            self.dp_camera['K'] = self.dp_camera['K'] * ssaa_fact
            self.camera_intrinsic_matrix = self.dp_camera['K']

            self.fx = self.camera_intrinsic_matrix[0, 0]
            self.fy = self.camera_intrinsic_matrix[1, 1]
            self.cx = self.camera_intrinsic_matrix[0, 2]
            self.cy = self.camera_intrinsic_matrix[1, 2]

            self.viewport_width  = im_size_rgb[0]
            self.viewport_height = im_size_rgb[1]

        elif(self.mode=="depth"):

            im_size = self.dp_camera['im_size']
            self.camera_intrinsic_matrix = self.dp_camera["K"]

            self.fx = self.camera_intrinsic_matrix[0, 0]
            self.fy = self.camera_intrinsic_matrix[1, 1]
            self.cx = self.camera_intrinsic_matrix[0, 2]
            self.cy = self.camera_intrinsic_matrix[1, 2]

            self.viewport_width  = im_size[0]
            self.viewport_height = im_size[1]


    def load_model(self, path_model):
        print("loading models ...",path_model)
        self.mesh_trimesh = trimesh.load(path_model)
        

        # import trimesh.visual as visual
        # # マテリアルを作成する
        # material = visual.material.SimpleMaterial(
        # #diffuse=[0.8, 0.2, 0.2],   # 赤みがかった拡散反射色
        # roughness=0.1,             # 中程度のラフネス
        # metallic=1.0,              # 非金属
        #   )

        # # マテリアルをメッシュに適用する
        # mesh_trimesh.visual.face_colors[:] = material.diffuse
        # mesh_trimesh.visual.material = material
        #model_mesh = pyrender.Mesh.from_trimesh(mesh_trimesh, smooth = True) # smoothgadame
        
        model_mesh = pyrender.Mesh.from_trimesh(self.mesh_trimesh)

        #model_mesh = pyrender.Mesh.from_trimesh(mesh)
        self.model_mesh = model_mesh

    def _set_scene(self, ambient_light=np.array([0.5, 0.5, 0.5,0.5])):
        """A hierarchical scene graph.

        Parameters
        ----------
        :param bg_color: Color of the background (R, G, B, A).
        :param ambient_light : (4,) float, optional
            Color of ambient light. Defaults to no ambient light.
        """
        self.scene = pyrender.Scene(bg_color=self._bg_color)
        #self.scene = pyrender.Scene(ambient_light=ambient_light, bg_color=self._bg_color)
        #self.add_lights()
        
    def set_light_on_scene(self, type="Direction", color=[1,1,1], intensity=10.0, light_pose=np.eye(4)):
        """
        type(str):光源タイプを決める
        color(List(int)):色を決める(0から1)
        intensity(float):明るさを決める
        """
        if(type=="Direction"):
            light = pyrender.DirectionalLight(color=color, intensity=intensity)
        elif(type=="Spot"):
            light =  pyrender.SpotLight(color=color, intensity=intensity,
                            innerConeAngle=np.pi/4, outerConeAngle=np.pi/2)
        elif(type=="Point"):
            light =  pyrender.PointLight(color=color, intensity=intensity)

        self.light_node  = self.scene.add(light, pose=light_pose)
        #self.light_nodes.append(light_node)

    def _fibonacci_sampling(self, n_points, radius=1.0):
        """Samples an odd number of almost equidistant 3D points from the Fibonacci
        lattice on a unit sphere.

        Latitude (elevation) represents the rotation angle around the X axis.
        Longitude (azimuth) represents the rotation angle around the Z axis.

        Ref:
        [1] https://arxiv.org/pdf/0912.4540.pdf
        [2] http://stackoverflow.com/questions/34302938/map-point-to-closest-point-on-fibonacci-lattice
        [3] http://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
        [4] https://www.openprocessing.org/sketch/41142
        change_camera_poseo sample (an odd number).
        :param radius: Radius of the sphere.
        :return: List of 3D points on the sphere surface.
        """
        # Needs to be an odd number [1].
        assert (n_points % 2 == 1)
        n_points_half = int(n_points / 2)

        phi = (np.sqrt(5.0) + 1.0) / 2.0  # Golden ratio.
        phi_inv = phi - 1.0
        ga = 2.0 * np.pi * phi_inv  # Complement to the golden angle.

        points = []
        for i in range(-n_points_half, n_points_half + 1):
            lat = np.arcsin((2 * i) / float(2 * n_points_half + 1))
            lon = (ga * i) % (2 * np.pi)

            # Convert the latitude and longitude angles to 3D coordinates.
            s = np.cos(lat) * radius
            x, y, z = np.cos(lon) * s, np.sin(lon) * s, np.tan(lat) * s
            points.append([x, y, z])

            # Calculate rotation matrix and translation vector.
            # Note: lat,lon=0,0 is a camera looking to the sphere center from
            # (-radius, 0, 0) in the world (i.e. sphere) coordinate system.
            # pi_half = 0.5 * np.pi
            # alpha_x = -lat - pi_half
            # alpha_z = lon + pi_half
            # R_x = transform.rotation_matrix(alpha_x, [1, 0, 0])[:3, :3]
            # R_z = transform.rotation_matrix(alpha_z, [0, 0, 1])[:3, :3]
            # R = np.linalg.inv(R_z.dot(R_x))
            # t = -R.dot(np.array([x, y, z]).reshape((3, 1)))

        return points
        
    def add_lights(self, num_lights=7, radius=100, intensity=150000.0):   
        num_lights=np.min([7,num_lights]) # 9okasikunaru
        
    
        points = self._fibonacci_sampling(num_lights, radius=radius)
          # 球状に配置された点光源を追加

        for point in points:
            # xy平面の角度 [0, 2π]
            azimuth = np.arctan2(point[1], point[0])
            if azimuth < 0:
                azimuth += 2.0 * np.pi

            # xy平面とz軸がなす角度 [-π/2, π/2]
            a = np.linalg.norm(point)
            b = np.linalg.norm([point[0], point[1], 0])
            elevation = np.arccos(b / a)
            if point[2] < 0:
                elevation = -elevation

            H_cam2w_lefthand = fix_camera_pose_generate(np.array([0,0,0]), azimuth, elevation, axial_angle=0, distance=radius)

            #light = pyrender.DirectionalLight(color=np.ones(3), intensity=intensity)
            #self.scene.add(light, pose=light_pose_right_hand)
            light =  pyrender.SpotLight(color=np.ones(3), intensity=intensity,
                            innerConeAngle=np.pi/8, outerConeAngle=np.pi/3)
                            
            self.scene.add(light, pose=H_cam2w_lefthand)
            
            
    def remove_lights_from_scene(self):
        if(len(self.light_nodes)>0):
            for i,light_node in enumerate(self.light_nodes):
                self.scene.remove_node(light_node)
                #del self.light_nodes[i]
        self.light_nodes = []

    def set_model_on_scene(self, model):
        self.model_node  = self.scene.add(model, pose=self._model_pose)

    def _set_camera_on_scene(self):
        # zfarがないと絵が出ない
        self._camera   = pyrender.IntrinsicsCamera(self.fx, self.fy, self.cx, self.cy, zfar=500)

        camera_p = pyrender.PerspectiveCamera(yfov=np.pi/3, aspectRatio=1.0)
        self.camera_node  = self.scene.add(self._camera, pose=self._model_pose)
        self.set_light_on_scene(type="Direction", intensity=1500.0, light_pose=self._model_pose)


    def remove_model_from_scene(self):
        if(self.model_node is not None):
            self.scene.remove_node(self.model_node)
            self.model_node = None

    def change_camera_pose(self, H_cam2w_rightand):
        self.scene.set_pose(self.camera_node, H_cam2w_rightand)
        #print("det H_cam2w_rightand",np.linalg.det(H_cam2w_rightand))
        self.scene.set_pose(self.light_node, H_cam2w_rightand)
        #self.scene.set_pose(self.light_node, self._convert_bw_left_and_right_hand(H_cam2w_rightand))# matigai 

    def _render_imgs(self):

        r = pyrender.OffscreenRenderer(viewport_width=self.viewport_width, viewport_height=self.viewport_height)
        

        rgb, depth = r.render(self.scene)#,pyrender.constants.RenderFlags.SHADOWS_ALL)

        depth = depth.astype(np.float32)
        mask  = (depth > 0).astype(np.uint8) * 255
        r.delete()
        return rgb, depth, mask

    def _convert_bw_left_and_right_hand(self, H): 
        """右手系と左手系を入れ替える(z軸をフリップするだけ)
        """
        H_copy = copy.copy(H)
        H_copy[:,2] = -H_copy[:,2]   
        return H_copy

    def compute_distance_difference(self, camera_pose, deth, mask):
        object_centroid_world = np.array([0, 0, 0])

        # Camera position in world coordinates
        camera_position = camera_pose[:3, 3]

        depth_non_zero = deth[mask!=0]
        thresh = np.percentile(depth_non_zero,30)
        distance_to_surface = np.median(depth_non_zero[depth_non_zero>thresh])

        # Compute distance from camera to object centroid
        distance_to_centroid = np.linalg.norm(camera_position - object_centroid_world)

        distance_difference = np.abs(distance_to_centroid - distance_to_surface)*0.001
        # print(distance_to_centroid,distance_to_surface,np.median(depth_non_zero))

        return distance_difference

        
    # def compute_distance_difference(self, camera_pose, mesh):
    #     # Compute object centroid and transform to world coordinates (origin)
    #     object_centroid_world = np.array([0, 0, 0])

    #     # Camera position in world coordinates
    #     camera_position = camera_pose[:3, 3]

    #     # Compute distance from camera to object centroid
    #     distance_to_centroid = np.linalg.norm(camera_position - object_centroid_world)

    #     # Compute ray direction from camera to object center (origin)
    #     ray_direction = -camera_position / np.linalg.norm(camera_position)

    #     # Perform ray-mesh intersection to find distance to surface
    #     ray_origins = np.array([camera_position])
    #     ray_directions = np.array([ray_direction])

    #     # Use RayMeshIntersector for ray-mesh intersection
    #     intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh, scale_to_box=False)
    #     intersections, index_ray = intersector.intersects_id(
    #         ray_origins=ray_origins, 
    #         ray_directions=ray_directions
    #     )

    #     if len(intersections) == 0:
    #         print("Ray did not intersect the mesh")
    #         return None

    #     # 最初の交差点を使用
    #     intersection_point = intersections[0]
    #     # Compute distance to the first intersection point
    #     distance_to_surface = np.linalg.norm(camera_position - intersection_point)

    #     # Compute the distance difference
    #     distance_difference = np.abs(distance_to_centroid - distance_to_surface)*0.001

    #     return distance_difference

def get_direction(azimuth, elevation):
    """極座標系で方向ベクトルを求める
    """
    x = np.cos(elevation) * np.cos(azimuth)
    y = np.cos(elevation) * np.sin(azimuth)
    z = np.sin(elevation)

    direction_vector =  np.array([x, y, z])
    direction_vector = direction_vector / np.linalg.norm(direction_vector)

    return direction_vector
 

def get_view_matrix(campos_w:list, tarpos_w:list, upvector:list): 
    """視野変換行列を求める 

    Args: 
        campos_w (list): world座標系上のカメラの位置(x,y,z) [m] 
        tarpos_w (list): world座標系上のカメラの視点の先(x,y,z) [m] 
        upvector (list): カメラの上方向ベクトル 
    """ 

    campos_w = np.array(campos_w) 
    tarpos_w = np.array(tarpos_w) 
    upvector = np.array(upvector) 

    # z 軸 = (l - c) / ||l - c|| 
    # z_cam = tarpos_w - campos_w 
    z_cam = -(tarpos_w - campos_w) 
    z_cam = z_cam / np.linalg.norm(z_cam) 


    # x 軸 = (z_cam × u) / ||z_cam × u|| 
    x_cam = np.cross(z_cam, upvector) 

    # upvectorとz_camがほぼ同じ向きの場合、x_camの外積の結果が0ベクトルになってしまう 
    if np.count_nonzero(x_cam) == 0: 
        upvector = np.array([1, 0, 0]) 
        x_cam = np.cross(z_cam, upvector) 

    x_cam = x_cam / np.linalg.norm(x_cam) 

    # y 軸 = (z_cam × x_cam) / ||z_cam × x_cam|| 
    y_cam = np.cross(z_cam, x_cam) 
    # y_cam = np.cross(x_cam, z_cam) 
    y_cam = y_cam / np.linalg.norm(y_cam) 


    tx = -np.dot(campos_w , x_cam) 
    ty = -np.dot(campos_w , y_cam) 
    tz = -np.dot(campos_w , z_cam) 
    
    cHw = np.array([[x_cam[0], x_cam[1], x_cam[2], tx], 
                   [ y_cam[0], y_cam[1], y_cam[2], ty], 
                   [ z_cam[0], z_cam[1], z_cam[2], tz], 
                   [        0,        0,        0,  1]]) 

    return cHw 



def fix_camera_pose_generate(look_at_pos, azimuth, elevation, axial_angle, distance):

    # 視点先位置
    look_at_pos = look_at_pos

    direction = get_direction(azimuth, elevation)

    # カメラ位置 = 視点先位置 + 大きさ * 方向ベクトル
    cameraPosition = look_at_pos + distance * direction

    H_w2cam = get_view_matrix(
        upvector=np.array([0, 0, 1]),
        campos_w=cameraPosition,
        tarpos_w=look_at_pos)

    # 視点軸周りに回転させる
    # 回転ベクトル
    rotation_vector = axial_angle * copy.deepcopy(direction)

    # ロドリゲスの回転公式で回転ベクトルを回転行列に変換
    R,_       = cv2.Rodrigues(rotation_vector)
    HR        = np.zeros((4,4))
    HR[:3,:3] = R
    HR[3,3]   = 1
    H_w2cam   = H_w2cam @ HR
    H_cam2w   = np.linalg.inv(H_w2cam)

    return H_cam2w


def generate_camera_poses(viewpoints,
                          look_at_pos,
                          axial_angles,
                          initial_distance_mm,
                          end_distance_mm,
                          distance_division_number,
                          azimuth_range=(0, 2 * np.pi),
                          elev_range=(-0.5 * np.pi, 0.5 * np.pi)):
    H_cam2w_lefthands = []

    distances  = np.linspace(initial_distance_mm,
                             end_distance_mm,
                             distance_division_number)

    for distance in distances:
        for viewpoint in viewpoints:

            # xy平面の角度
            azimuth = np.arctan2(viewpoint[1], viewpoint[0])
            if azimuth < 0:
                azimuth += 2.0 * np.pi

            # xy平面とz軸がなす角度
            a = np.linalg.norm(viewpoint)
            b = np.linalg.norm([viewpoint[0], viewpoint[1], 0])
            elevation = np.arccos(b / a)
            if viewpoint[2] < 0:
                elevation = -elevation

            if not (azimuth_range[0] <= azimuth <= azimuth_range[1] and
                    elev_range[0] <= elevation <= elev_range[1]):
                continue

            for axial_angle in axial_angles:
                H_cam2w_lefthand = fix_camera_pose_generate(look_at_pos, azimuth, elevation, axial_angle, distance)
                H_cam2w_lefthands.append(H_cam2w_lefthand)

    return H_cam2w_lefthands


def standard_render():
    parser = config_parser()
    args = parser.parse_args()

    # 使用例
    input_paths = RenderedImageInputPath(args.path_camera_intrinsic_param, args.path_camera_coordinates, args.mesh)
    output_paths = RenderedImageOutputPath(output_dir=args.output_dir, result_dirname=args.result_dirname, obj_id=input_paths.obj_id)

    with open(output_paths.out_profile_tpath, 'w') as f:
        json.dump(vars(args), f)

    print("*"*99)
    print(f'{input_paths.obj_id}')
    start_time = time.time()  # 処理の開始時間を記録

    renderer = Render(input_paths=input_paths,
                      output_paths=output_paths,
                      axial_angle_division_number=args.axial_angle_division_number,
                      bg_color=args.background_color)

    renderer.render()

    end_time = time.time()  # 処理の終了時間を記録
    elapsed_time = end_time - start_time  # 経過時間を計算
    print("Rendering Time: {:.2f} seconds".format(elapsed_time))  # 経過時間を表示
    print("*"*99)


if __name__== "__main__":
    standard_render()
