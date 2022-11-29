# TALMISER

Here's the code for project of CMPUT622.

## Running Steps

### Dependencies

To install the dependecies, use

        pip install -r requirements.txt

Tips: Please use `python>=3.8`, since there are some walrus operators, which is the feature of Python 3.8.

### Training

        python train.py --dataset <dataset_name> --model <model_name>
or in short

        python train.py -d <dataset_name> -m <model_name>

### Generating DTMC

        python DTMC_generator.py --dataset census --model DNN --attribute <attribution_name> --plot --epsilon <epsilon> --delta <delta>

or in short with default $\epsilon=0.01$, $\delta=0.05$

        python DTMC_generator.py -d <dataset_name> -m <model_name> -a <attribution_name> --plot

If you don't want to output the DTMC image, just drop the option `--plot`

If you‘d like to generate DTMC in origin method, please running with option `--o ori`


### Verification

        python verification.py --dataset <dataset_name> --model <model_name> --fair_diff <available_diff_value>

or in short

        python verification.py -d <dataset_name> -m <model_name> -fd <available_diff_value>

## Inplement a new dataset or new model

All the config are stored in `configs/<dataset_name>.yaml`, you can add a new yaml file to add a new dataset, or add new model details in existing yaml file to build a new model.

After that, just repeat the running steps.
