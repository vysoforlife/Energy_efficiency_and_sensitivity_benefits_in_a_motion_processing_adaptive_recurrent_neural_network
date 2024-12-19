# Energy efficiency and sensitivity benefits in a motion processing adaptive recurrent neural network

The Python code in this repository serves as a Key Resource for:

**Mohan & Rideaux (2024) Energy efficiency and sensitivity benefits in a motion processing adaptive recurrent neural network**

This code can be used to train and test the two recurrent neural network models—**MotionNet-R** (baseline, non-adaptive) and **AdaptNet** (adaptive)—to reproduce the results presented in the manuscript.

The training datasets (natural image sequences), as well as pre-trained networks and results files, can be found at: https://osf.io/pg7se/files/osfstorage/675bfc0197940e6cff87bbe1

---

## Instructions

### Training the networks

You can use the pre-trained models provided in the `Trained_Models` folder, or you can train new instances of MotionNet-R or AdaptNet from scratch.

**Core components:**

- **Training scripts**
  - `AdaptNet.py` : Defines the adaptive network architecture
  - `MotionNet-R.py` : Defines the baseline recurrent network architecture
  - `Combined_Training.py` : Provides a top-level script to train both MotionNet-R and AdaptNet

**To train a new model**, ensure that the training image sequences are placed in a peer folder named `Training_Data` (or as indicated by the script parameters), located in the parent directory. Then, run `Combined_Training.py` to generate both model types (MotionNet-R and AdaptNet).

Training duration will depend on your system’s computational resources, although it is recommended to use an nvidia GPU with CUDA enabled, for the same.

---

### Testing the networks

If you have valid pre-trained instances of MotionNet-R or AdaptNet (e.g., 10 trained models per network type, as indicated in the paper) stored in the `Trained_Models` directory, you can evaluate their performance and reproduce the figures and analyses in the manuscript.

**Core components:**

- **Testing scripts** (in `Testing_Scripts` directory):
  - `Polar_Plots.py` : Reproduces figures related to sine and plaid responses
  - `Efficiency_Bar_Graphs.py` : Recreates bar plots comparing energy use and accuracy
  - `Sensitivity_Analysis.py` : Tests sensitivity of networks to changes in motion
  - `RNN_Activity.py` : Evaluates the networks for motion aftereffect-like responses and plots RNN activity to those stimuli as well
  - `Model_Weights.py` : Visualizes and analyzes learned weights
  - `Latency_Analysis.py` : Compares latency of both network types to changes in the input stimulus velocity

**To test a model**, ensure that pre-trained model files are available in the `Trained_Models` directory. Run the desired testing scripts to compute the estimates or other properties of the model. After processing is complete, any results will be saved to the `Saved_Images` directory, where you can visualize outcomes similar to those reported in the paper.

---

### Visualization and results

Many testing scripts produce figures or data logs used to generate the final figures in the paper. These plots are saved in `Saved_Images`. For example:

- `Polar_Plots.py` generates responses of MotionNet-R to sine and plaid stimuli (cf. Figures related to V1/MT tuning).
- `Efficiency_Bar_Graphs.py` creates comparisons of RNN output and MSE (cf. efficiency results in the paper).
- `Sensitivity_Analysis.py` outputs data that can be visualized as sensitivity matrices and fitted sigmoid curves, demonstrating improved sensitivity to changing motion conditions for AdaptNet.

By following these steps and running the respective scripts, you can closely replicate the main findings presented in the manuscript.

---

**Contact & Acknowledgements**

For queries about the code and analyses, please contact: reuben.rideaux@sydney.edu.au

Please see the manuscript for full acknowledgements and references.
