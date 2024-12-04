import numpy as np
from typing import Optional,List
import copy

class ReflectionPlane:
    def __init__(self, *, plane_pos: Optional[np.ndarray] = None, 
                 plane_normal: Optional[np.ndarray] = None, 
                 plane_parameter: Optional[List[float]] = None, 
                 reflection_matrix: Optional[np.ndarray] = None) -> None:
        """
        ReflectionPlaneクラスのコンストラクタ。

        Parameters:
            plane_pos (Optional[np.ndarray]): 反射平面上の点の位置ベクトル。
            plane_normal (Optional[np.ndarray]): 反射平面の法線ベクトル。
            plane_parameter (Optional[List[float]]): 反射平面のパラメータ[a, b, c, d]。
            reflection_matrix (Optional[np.ndarray]): 反射行列。

        Raises:
            ValueError: 引数が省略された場合に発生します。
        """
        if reflection_matrix is not None:
            self._reflection_matrix = np.array(reflection_matrix)
            self._plane_normal = self._reflection_matrix2plane_pos_normalvec()
            # 注意：
            # reflection_matrix -> plane_posはできない, plane_normal はできる
            # reflection_matrix -> plane_parameterはできない
            # 対策：calculate_plane_posメソッド呼んだ後ならself._plane_posが設定されるのでdが計算できる -> 上の2つが計算できる
            self._plane_pos = None
            self._plane_parameter = None

        elif plane_parameter is not None:
            # 注意：
            # plane_parameter -> plane_posはできない, plane_normal はできる
            # plane_parameter -> reflection_matrixはできる
            self._plane_parameter = plane_parameter
            self._reflection_matrix = self._plane_params2reflection_matrix()
            self._plane_normal = self._plane_params2plane_pos_normalvec()
            # 対策：calculate_plane_posメソッド呼んだ後ならself._plane_posが設定されるのでdが計算できる 
            #       代わりにcalculate_plane_posメソッドで興味のある点を平面に写像して求める
            self._plane_pos = None

        elif plane_pos is not None and plane_normal is not None:
            # 注意：
            # (plane_pos,plane_normal) -> reflection_matrixはできる
            # (plane_pos,plane_normal) -> plane_parameterはできる
            self._plane_pos = np.array(plane_pos)
            self._plane_normal = np.array(plane_normal)
            self._reflection_matrix = self._plane_pos_normalvec2reflection_matrix()
            self._plane_parameter = self._plane_pos_normalvec2plane_parameter()

    @property
    def plane_pos(self) -> np.ndarray:
        """
        反射平面上の点の位置ベクトルを取得します。
        """
        if self._plane_pos is None:
            print("please call method 'calculate_plane_pos' instead.")
        return self._plane_pos

    @property
    def plane_normal(self) -> np.ndarray:
        """
        反射平面の法線ベクトルを取得します。
        """
        return self._plane_normal

    @property
    def plane_parameter(self) -> List[float]:
        """
        反射平面のパラメータ[a, b, c, d]を取得します。
        """
        if self._plane_parameter is None:
            # plane_parameterが設定されていなくてもcalculate_plane_posメソッド呼んだ後ならself._plane_posが設定されている
            if self._plane_pos is not None:
                # self._plane_posが設定されている -> plane_parameterが計算できる
                self._plane_parameter = self._plane_pos_normalvec2plane_parameter()
        return self._plane_parameter

    @property
    def reflection_matrix(self) -> np.ndarray:
        """
        反射行列を取得します。
        """
        return self._reflection_matrix
    
    def calculate_plane_pos(self, target_points: np.ndarray) -> np.ndarray:
        """
        反射平面上の点の位置ベクトルを計算します。

        注意:
            plane_pos は反射平面に対して垂直な座標からの距離を表しますが、
            平面上の空間の座標は与えられた点の座標の平均位置となりますので、注意してください。

        Parameters:
            target_points (numpy.ndarray): 反射面に対する各点の座標を表す配列。

        Returns:
            numpy.ndarray: 反射平面上の点の位置ベクトル。
        """
        p_homogeneous = np.hstack((target_points, np.ones((len(target_points), 1))))  # 同次座標系に変換
        q_homogeneous = p_homogeneous @ self._reflection_matrix.T  # 変換後の点を計算
        p = p_homogeneous[:, :3]  # 同次座標系から通常の座標系に変換
        q = q_homogeneous[:, :3]  # 同次座標系から通常の座標系に変換
        self._plane_pos = np.mean(0.5 * (p + q), axis=0)
        return self._plane_pos

    def _plane_pos_normalvec2reflection_matrix(self) -> np.ndarray:
        """
        平面の位置ベクトルと法線ベクトルから反射行列を計算します。
        """
        plane_normal = copy.deepcopy(self._plane_normal)
        plane_pos = copy.deepcopy(self._plane_pos)
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        R = np.eye(3) - 2 * np.outer(plane_normal, plane_normal)
        h = np.dot(plane_pos, plane_normal)
        t = 2 * h * plane_normal
        reflection_matrix = np.eye(4)
        reflection_matrix[:3, :3] = R
        reflection_matrix[:3, 3] = t.T
        return reflection_matrix

    def _plane_pos_normalvec2plane_parameter(self) -> List[float]:
        """
        平面の位置ベクトルと法線ベクトルから平面のパラメータ[a, b, c, d]を計算します。

        Returns:
            list: 平面のパラメータ[a, b, c, d]。
        """
        a, b, c = self._plane_normal
        d = -np.dot(self._plane_normal, self._plane_pos)
        return [a, b, c, d]
    
    def _plane_params2plane_pos_normalvec(self) -> np.ndarray:
        """
        平面のパラメータ[a, b, c, d]から平面の位置ベクトルと法線ベクトルを計算します。

        Returns:
            list: 平面の位置ベクトルと法線ベクトルのリスト。
        """
        return self._plane_parameter[:3]

    def _plane_params2reflection_matrix(self) -> np.ndarray:
        """
        平面のパラメータ[a, b, c, d]から反射行列を計算します。
        """
        a, b, c, d = self._plane_parameter
        plane_normal = np.array([a, b, c])
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        t = -2 * d * plane_normal
        R = np.eye(3) - 2 * np.outer(plane_normal, plane_normal)
        reflection_matrix = np.eye(4)
        reflection_matrix[:3, :3] = R
        reflection_matrix[:3, 3] = t
        return reflection_matrix
    
    def _reflection_matrix2plane_pos_normalvec(self) -> np.ndarray:
        """
        反射行列から平面の法線ベクトルを抽出します。
        """
        reflection_matrix = np.array(self._reflection_matrix)
        reflection_matrix_only_rot = reflection_matrix[:3, :3]
        eigenvalues, eigenvectors = np.linalg.eig(reflection_matrix_only_rot)
        symmetry_idx = np.argmin(np.abs(eigenvalues + 1))
        if np.abs(eigenvalues[symmetry_idx] + 1) >= 1e-6:
            raise ValueError('no -1 eigenvalue')
        plane_normal = eigenvectors[:, symmetry_idx].real
        return plane_normal

    def _plane_pos_normalvec2reflection_matrix(self) -> np.ndarray:
        """
        平面の法線ベクトルと位置ベクトルから反射行列を計算します。
        """
        a, b, c = self._plane_normal
        d = -np.dot(self._plane_normal, self._plane_pos)
        plane_normal = np.array([a, b, c])
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        t = -2 * d * plane_normal
        R = np.eye(3) - 2 * np.outer(plane_normal, plane_normal)
        reflection_matrix = np.eye(4)
        reflection_matrix[:3, :3] = R
        reflection_matrix[:3, 3] = t
        return reflection_matrix