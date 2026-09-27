# Custom processing block example (Python)

This is an example of a custom processing block, which you can load in the Edge Impulse studio. See the docs: [Building custom processing blocks](https://docs.edgeimpulse.com/docs/custom-blocks).

1. Install Python 3.12:
2. Create a venv, and install packages (note: you cannot update these, or add new packages - this is a fixed package list for Edge Impulse DSP blocks):

    ```bash
    python3.12 -m venv .venv
    source .venv/bin/activate
    pip3 install -r requirements.txt
    ```
