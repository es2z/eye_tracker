"""視線オーバーレイ for OBS"""

import os

# cv2とmediapipeより先に読み込まれるよう、ここで環境変数を設定する。
# MSMFのハードウェア変換を切ると、カメラを開く時間が7〜8秒から0.3秒程度に短くなる(実測)
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
# OpenCVのWARNログ(バックエンド探索時のもの)を抑える
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
# mediapipe(glog)のINFO/WARNINGログを抑える
os.environ.setdefault("GLOG_minloglevel", "2")

__version__ = "0.1.0"
