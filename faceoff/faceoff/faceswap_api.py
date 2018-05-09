from lib.utils import FullHelpArgumentParser
from scripts.extract import ExtractTrainingData


class FaceSwapInterface:
  def __init__(self):
    self._parser = FullHelpArgumentParser()
    self._subparser = self._parser.add_subparsers()

  def extract(self, input_dir, output_dir, filter_path, processes):
    extract = ExtractTrainingData(
      self._subparser, "extract", "Extract the faces from a pictures.")
    args_str = "extract --input-dir {} --output-dir {} --processes {} --detector cnn --filter {}"
    args_str = args_str.format(input_dir, output_dir, processes, filter_path)
    self._run_script(args_str)

  def train(self, input_a_dir, input_b_dir, model_dir, gan = False):
    model_type = "Original"
    if gan:
      model_type = "GAN"
    train = TrainingProcessor(
      self._subparser, "train", "This command trains the model for the two faces A and B.")
    args_str = "train --input-A {} --input-B {} --model-dir {} --trainer {} --batch-size {} --write-image"
    args_str = args_str.format(input_a_dir, input_b_dir, model_dir, model_type, 512)
    self._run_script(args_str)

  def _run_script(self, args_str):
    args = self._parser.parse_args(args_str.split(' '))
    args.func(args)
