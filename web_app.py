#!/usr/bin/env python
import os
import pandas as pd
from argparse import ArgumentParser
from web_app import app


def create_form_opts():
    """
    Create list of test images
    for a dropdown menu.
    """

    test_images = pd.read_csv(os.path.join(os.getcwd(), "eda/test.csv"))
    test_images = test_images[["id"]]
    test_images.loc[:, "img_name"] = test_images.loc[:, "id"].apply(
        lambda x: "%05d.jpg" % x
    )
    form_opts = test_images[["img_name"]]
    base_path = os.path.join(os.getcwd(), "web_app/static/textdata")
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    form_opts_path = os.path.join(base_path, "form_options.csv")
    form_opts.to_csv(form_opts_path, index=False)


if __name__ == "__main__":

    # create options for a dropdown menu
    create_form_opts()

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--endpoint_name", type=str, default="")
    parser.add_argument(
        "--lambda_function_name", type=str, default="serve-cv-object-counter"
    )
    parser.add_argument("--s3_bucket", type=str, default="amazon-bin-images-sub")
    parser.add_argument("--s3_prefix", type=str, default="")
    args = parser.parse_args()

    app.config.update(
        endpoint_name=args.endpoint_name,
        lambda_function_name=args.lambda_function_name,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
    )

    # run app
    app.run(host="0.0.0.0", port="80")
