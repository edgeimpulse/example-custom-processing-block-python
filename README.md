# Custom processing block example (Python)

This is an example of a custom processing block, which you can load in the Edge Impulse studio. See the docs: [Building custom processing blocks](https://docs.edgeimpulse.com/docs/custom-blocks).

1. Install Python 3.12:
2. Create a venv, and install packages (note: you cannot update these, or add new packages - this is a fixed package list for Edge Impulse DSP blocks):

    ```bash
    python3.12 -m venv .venv
    source .venv/bin/activate
    pip3 install -r requirements.txt
    ```

3. In Edge Impulse, go to a DSP page; copy the raw features for a sample, and place in `features.txt`:

    ![Copy features](images/copy-features.png)

4. Run the DSP block:

    ```
    python3 run.py --features features.txt --frequency 62.5 --axes "accX,accY,accZ" --scale-axes 1
    ```

## Adding extra parameters

If you have new parameters you want to add to the block:

1. Add them as arguments to `generate_features` in [dsp.py](dsp.py).
2. Also add them to the `parameters` section in `parameters.json`. This will ensure there's UI rendered to configure the new parameters. See the `DSPParameterItem` spec in https://docs.edgeimpulse.com/tools/specifications/files/parameters-json for all options.

`run.py` will automatically pick up new parameters in `parameters.json`.
