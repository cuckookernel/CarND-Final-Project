
import tensorflow as tf
from train_tl_classifier import ARCH_3_3_TL


class TLClassifierDL(object):
    def __init__(self, sess, ckpt_path, img_wh):
        """Builds a tensorflow graph and restores parameters from a checkpoint file"""
        # tf.reset_default_graph()
        self.img_wh = img_wh
        print( type(self.img_wh) )
        hyp_pars = dict(learning_rate=0.0005, batch_size=64, keep_prob=0.6)
        X_shape = ( img_wh[1], img_wh[0], 3 )
        self.tnsr = h.build_network_and_metrics( X_shape, 4, ARCH_3_3_TL, hyp_pars)
        saver = tf.train.Saver()
        saver.restore( sess, ckpt_path )

    def get_classification(self, sess, img0):

        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # print( img0.shape, img0.dtype )
        img = cv2.resize(img0, self.img_wh )
        img = cv2.cvtColor( img, cv2.COLOR_BGR2HLS )
        img1 = ( (img - img.mean() )/ img.std() ).reshape( tuple( [1] + list(img.shape) ) )

        tnsr = self.tnsr

        logits = sess.run(tnsr["logits"], feed_dict={tnsr["input"]: img1,
                                                     tnsr['keep_prob']: 1.})[0]


        pred_class = np.argmax( logits )

        return (pred_class, logits) if pred_class < 3 else (4, logits)