import os
import sys
import tensorflow as tf

try:
    from Model.UNetModel import AudioTrackSeparation
except ImportError:
    from UNetModel import AudioTrackSeparation

def test_model(config, test_input):
    '''
        Args:
           config: configuration variable to get model weight path.
           test_input: Processed array of provided audio data.
        Returns: Model prediction for processed input array.
    '''
    model = AudioTrackSeparation()
    model.build(input_shape=(None, config.dim, 1))

    weights_path = os.path.join(config.model_save_path, 'model_weights')
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if not (os.path.exists(weights_path + '.index') or os.path.exists(weights_path)):
        # Check relative to script directory
        alt_save_path = os.path.join(script_dir, config.model_save_path)
        alt_weights = os.path.join(alt_save_path, 'model_weights')
        if os.path.exists(alt_weights + '.index') or os.path.exists(alt_weights):
            weights_path = alt_weights
        elif os.path.exists(config.model_save_path):
            latest_ckpt = tf.train.latest_checkpoint(config.model_save_path)
            if latest_ckpt:
                weights_path = latest_ckpt
        elif os.path.exists(alt_save_path):
            latest_ckpt = tf.train.latest_checkpoint(alt_save_path)
            if latest_ckpt:
                weights_path = latest_ckpt

    print(f"Loading weights from: {weights_path}")
    model.load_weights(weights_path).expect_partial()

    test_tensor = tf.convert_to_tensor(test_input, dtype=tf.float32)
    pred_vocals, pred_drums = model(test_tensor, training=False)

    return [pred_vocals.numpy(), pred_drums.numpy()]