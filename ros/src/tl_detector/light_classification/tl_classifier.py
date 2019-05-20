# from styx_msgs.msg import TrafficLight


class TLClassifier(object):
    def __init__(self, sess, ckpt_path, img_wh):
        """Builds a tensorflow graph and restores parameters from a checkpoint file"""
        # tf.reset_default_graph()
        pass

    def get_classification(self, ignored, img):
        red_pxs = (img[:, :, 0] < 70) & (img[:, :, 1] < 70) & (img[:, :, 2] > 230)

        return 0 if red_pxs.sum() > 90 else 4

