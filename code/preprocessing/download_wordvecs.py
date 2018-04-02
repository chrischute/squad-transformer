# Copyright 2018 Christopher Chute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Download GloVe Common Crawl vectors and unzip them."""

import argparse
import os
import zipfile
from squad_preprocess import maybe_download


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    return parser.parse_args()


def main():
    args = setup_args()
    glove_base_url = "http://nlp.stanford.edu/data/"
    glove_filename = "glove.840B.300d.zip"

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir, exist_ok=True)

    maybe_download(glove_base_url, glove_filename, args.data_dir)

    glove_path = os.path.join(args.data_dir, glove_filename)
    print("Unzipping {}...".format(glove_path))
    glove_zip_ref = zipfile.ZipFile(glove_path, 'r')

    glove_zip_ref.extractall(args.data_dir)
    glove_zip_ref.close()


if __name__ == '__main__':
    main()
