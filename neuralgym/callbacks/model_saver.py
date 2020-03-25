"""model_saver"""
import os

from . import PeriodicCallback, CallbackLoc
from ..utils.logger import callback_log


import subprocess



class ModelSaver(PeriodicCallback):
    """Save model to file at every pstep step_start.

    Args:
        pstep (int): Save to model every pstep.
        saver: Tensorflow saver.
        dump_prefix (str): Prefix for saving model files.
        gcloud_bucket_path: String specifying path to Google cloud bucket.

    """

    def __init__(self, pstep, saver, dump_prefix, gcloud_bucket_path=None):
        super().__init__(CallbackLoc.step_start, pstep)
        self._saver = saver
        self._dump_prefix = dump_prefix

        self._gcloud_bucket_path = gcloud_bucket_path  # Path to bucket.

        dump_dir = os.path.dirname(self._dump_prefix)
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
            callback_log('Initialize ModelSaver: mkdirs %s.' % dump_dir)

    def run(self, sess, step):
        if step != 0:
            callback_log('Trigger ModelSaver: Save model to {}-{}.'.format(
                self._dump_prefix, step))
            self._saver.save(sess, self._dump_prefix, global_step=step)

            if self._gcloud_bucket_path is not None:
                # When checkpointing, multiple files are created at this 'step',
                # so we wildcard them to copy them in one go to the bucket.
                #
                file_to_copy = self._dump_prefix + '-' + str(step) + '*'

                # Where the checkpoint files will be stored in Google cloud.
                #
                bucket_path = self._gcloud_bucket_path + "/models/"

                callback_log('Trigger ModelSaverGoogleCloud: Copying model to {}->{}.'.format(
                    self._dump_prefix, bucket_path))

                subprocess.check_call([
                    'gsutil', 'cp',
                    file_to_copy,
                    bucket_path
                ])
