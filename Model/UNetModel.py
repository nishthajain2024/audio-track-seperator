from Model.UNetBlocks import DownscaleBlock, UpscaleBlock, BottleNeckBlock
import tensorflow as tf
from tensorflow.keras import layers

from config import get_parameters

class AudioTrackSeparation(tf.keras.Model):
    def __init__(self):
        super().__init__()
        f = [8, 16, 32, 64, 128, 256]
        
        self.downscale_blocks = [
            DownscaleBlock(f[0]),
            DownscaleBlock(f[1]),
            DownscaleBlock(f[2]),
            DownscaleBlock(f[3]),
            DownscaleBlock(f[4]),
        ]

        self.bottle_neck_block = BottleNeckBlock(f[5])

        self.upscale_blocks = [
            UpscaleBlock(f[4]),
            UpscaleBlock(f[3]),
            UpscaleBlock(f[2]),
            UpscaleBlock(f[1]),
            UpscaleBlock(f[0]),
        ]

        self.output_layer_0 = layers.Conv1D(1, 1, 1, activation='tanh', 
                                            padding='valid', name='output_0')
        self.output_layer_1 = layers.Conv1D(1, 1, 1, activation='tanh', 
                                            padding='valid', name='output_1')

        self.loss_metric_vocals = tf.keras.metrics.Mean(name="vocal_loss")
        self.loss_metric_drums = tf.keras.metrics.Mean(name="drums_loss")
        self.loss_metric_total = tf.keras.metrics.Mean(name="total_loss")
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    @property
    def metrics(self):
        return [self.loss_metric_vocals, self.loss_metric_drums, self.loss_metric_total]
    
    def compile(self, 
                optimizer=None,
                vocals_optimizer=None,
                drums_optimizer=None,
                **kwargs):
        super().compile(**kwargs)
        if optimizer is not None:
            self.optimizer = optimizer
        elif vocals_optimizer is not None:
            self.optimizer = vocals_optimizer

    def calculate_loss(self, target, pred):
        '''
        Optimizes balanced L1 Loss + Mean Squared Error without distortion.
        '''
        l1_loss = tf.reduce_mean(tf.abs(target - pred))
        mse_loss = tf.reduce_mean(tf.square(target - pred))
        return l1_loss + mse_loss
        
    def train_step(self, batch_data):
        input_data, target = batch_data
        target0, target1 = target

        with tf.GradientTape() as tape:
            pred0, pred1 = self(input_data, training=True)

            loss0 = self.calculate_loss(target0, pred0)
            loss1 = self.calculate_loss(target1, pred1)
            total_loss = loss0 + loss1

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_metric_vocals.update_state(loss0)
        self.loss_metric_drums.update_state(loss1)
        self.loss_metric_total.update_state(total_loss)

        return {
            "vocal_loss": self.loss_metric_vocals.result(),
            "drums_loss": self.loss_metric_drums.result(),
            "total_loss": self.loss_metric_total.result()
        }
    
    def test_step(self, batch_data):
        input_data, target = batch_data
        target0, target1 = target

        pred0, pred1 = self(input_data, training=False)

        loss0 = self.calculate_loss(target0, pred0)
        loss1 = self.calculate_loss(target1, pred1)
        total_loss = loss0 + loss1
        
        self.loss_metric_vocals.update_state(loss0)
        self.loss_metric_drums.update_state(loss1)
        self.loss_metric_total.update_state(total_loss)

        return {
            "vocal_loss": self.loss_metric_vocals.result(),
            "drums_loss": self.loss_metric_drums.result(),
            "total_loss": self.loss_metric_total.result()
        }

    def build_graph(self):
        '''
           To build model graph and plot the model summary.
        '''
        config = get_parameters()
        x = layers.Input(shape=(config.dim, 1))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
        
    def call(self, x):
        c1, p1 = self.downscale_blocks[0](x)
        c2, p2 = self.downscale_blocks[1](p1)
        c3, p3 = self.downscale_blocks[2](p2)
        c4, p4 = self.downscale_blocks[3](p3)
        c5, p5 = self.downscale_blocks[4](p4)

        bn = self.bottle_neck_block(p5)

        u1 = self.upscale_blocks[0](bn, c5)
        u2 = self.upscale_blocks[1](u1, c4)
        u3 = self.upscale_blocks[2](u2, c3)
        u4 = self.upscale_blocks[3](u3, c2)
        u5 = self.upscale_blocks[4](u4, c1)

        return self.output_layer_0(u5), self.output_layer_1(u5)