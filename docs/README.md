# Technical Documentation: ReelDetect

This document provides a detailed technical overview of the ReelDetect project, including our unique approach, the market gap it addresses, system architecture, experimental results, and future directions.

---

## 1. Our Approach & Unique Value Proposition

The core challenge of this project is to differentiate between high-priority video (reels) and lower-priority content (feeds) within a single, encrypted data stream from a Social Networking Service (SNS) application. Our approach is founded on the principle of **privacy-preserving statistical fingerprinting**.

Instead of attempting to decrypt the traffic, ReelDetect analyzes the rich metadata and behavioral patterns of the encrypted flows. We hypothesized that the "rhythm and shape" of video traffic would be fundamentally different from that of feed traffic.

#### What Makes Our Solution Unique?

Our solution is innovative because it stands at the intersection of three critical, high-priority research areas, providing a solution that is simultaneously:

* **Fine-Grained:** Unlike traditional methods that only identify the parent application, our model performs in-app classification, differentiating between content types *within* the same application.
* **Lightweight & Efficient:** By choosing an optimized GBDT model (LightGBM) over more complex Deep Learning architectures, our solution is designed for on-device (Edge AI) deployment, proven to be **2.2x faster** and **8x smaller** than a baseline CNN.
* **Robust:** The model is not brittle. Through a deliberate data augmentation strategy, it is trained to be resilient to real-world network impairments like packet loss and latency, maintaining **98.7% accuracy**.

---

## 2. Market Gap Analysis

Our research into the current state of network traffic management reveals a significant and growing market gap that ReelDetect is perfectly positioned to fill.

#### The Encryption Wall & Lack of Granularity
The universal adoption of **TLS 1.3 and QUIC** has made traditional Deep Packet Inspection (DPI) obsolete. Existing solutions can typically only identify the domain (e.g., `*.youtube.com`), but they are blind to the content within. They cannot tell the difference between a user watching a 4K video and a user browsing comments, yet these activities have vastly different network requirements.

#### Reactive vs. Proactive QoS
Most current systems are **reactive**. They detect network strain (like high bandwidth usage) only *after* it has already begun, at which point a user's video may already be buffering. They are also inefficient, as they can't distinguish between a high-priority video stream and a low-priority background file download, potentially wasting premium network resources on non-critical traffic.

ReelDetect addresses this gap by being **proactive, granular, and privacy-preserving**. It intelligently identifies the *intent* to watch a video from the very first packets, allowing a device to proactively signal for better service *before* buffering occurs, all without ever compromising the privacy of the encrypted data.

---

## 3. Hypothesis Validation & Exploratory Data Analysis

Before building our model, we performed a rigorous Exploratory Data Analysis (EDA) to test our foundational hypotheses. The results provided a clear, data-backed justification for our approach.

![Violin Plot of Mean Packet Size](/src/output/plots/hypothesis_1_violin_plot_augmented.png)

The most critical finding came from analyzing the distribution of the largest packets (the 90th percentile), which revealed a distinct "fingerprint" for each traffic type.

![Distribution Plot of P90 Packet Size](/src/output/plots/hypothesis_2_kde_plot_augmented.png)

**Analysis:** The plot reveals that **Reel traffic (blue)** has a **bimodal distribution** (two peaks), representing large video segment downloads and small control packets. In contrast, **Non-Reel traffic (orange)** has a **unimodal distribution** (one peak). This undeniable structural difference proved that a rich, complex, and highly predictive pattern exists within the encrypted traffic.

---

## 4. Technical Stack

Our solution is a complete, end-to-end machine learning pipeline that transforms raw network packets into real-time, actionable insights.

`(Data Collection: Wireshark)` **->** `(Feature Extraction: process_capture.py)` **->** `(Data Augmentation: augment_data.py)` **->** `(Model Training: train_model.py)` **->** `(Live Inference: dashboard.py)`

1.  **Data Collection:** We captured live traffic from SNS apps in `.pcapng` format.
2.  **Feature Extraction:** A Python script processes these captures, grouping packets into flows and analyzing them in 3-second windows to generate a statistical feature vector.
3.  **Data Augmentation:** To ensure robustness, we programmatically inject noise into our clean dataset to simulate latency and packet loss, creating a larger, more diverse training set.
4.  **Model Training:** A LightGBM classifier is trained on the augmented dataset, learning the complex patterns that differentiate the traffic types.
5.  **Live Inference:** A Streamlit dashboard uses the trained model to classify live network traffic in real-time.

---

## 5. Technical Architecture of Our Solution

Our solution is a complete, end-to-end machine learning pipeline that transforms raw network packets into real-time, actionable insights.

`(Data Collection: Wireshark)` **->** `(Feature Extraction: process_capture.py)` **->** `(Data Augmentation: augment_data.py)` **->** `(Model Training: train_model.py)` **->** `(Live Inference: dashboard.py)`


---

## 6. Implementation Details

### 6.1 Feature Engineering
The success of our model hinges on a set of 11 carefully engineered features that form a "fingerprint" of the traffic's behavior. Key features include `downlink_throughput_bps`, `psz_p90_down`, and `burst_cnt`.

### 6.2 The Machine Learning Model & Experimental Results

Our journey to the final model was a process of iterative refinement and rigorous experimentation.

#### Experiment 1: Granularity & Accuracy
After fixing an initial data leakage issue, we trained our final robust model on an augmented dataset. It achieved an outstanding **98.7% accuracy**.

![Final Model Accuracy](/src/output/plots/train_model_output.png)
*(**Action:** Take a screenshot of the terminal output after running `src/train_model.py` and save it as `train_model_output.png` in the `docs/assets` folder.)*

#### Experiment 2: On-Device Efficiency
We benchmarked our LightGBM model against a baseline 1D-CNN. The results prove our model is the correct choice for an on-device, low-latency application.

![Benchmark Results](/src/output/plots/benchmark_speed.jpg)


#### Experiment 3: Robustness by Design
Our final experiment proved the value of our data augmentation. The plot below shows that as simulated packet loss increases, the accuracy of a "naive" model is inconsistent, while our "Robust Model" maintains near-perfect performance.

![Robustness Comparison Plot](/src/output/plots/robustness_comparison.png)

### 6.3 The Real-Time Dashboard (UI/UX)
To demonstrate our solution, we built an interactive dashboard using Streamlit. It features a multi-threaded architecture to ensure the UI remains responsive while the packet capture runs in the background.

![Dashboard Screenshot](/src/output/plots/dashboard.png)

---

## 7. Future Evolution of the Project

While ReelDetect is a powerful proof-of-concept, it also serves as a foundation for several exciting future research and development directions:

* **Multi-Class "Open-World" Classifier:** The current model is binary (Reel vs. Non-Reel). The next evolution would be a multi-class model that can identify other in-app activities like **Direct Messaging, Stories, or Live Streams**. Critically, this would involve tackling the "open-world problem" by training the model to classify traffic that doesn't match any known activity as "Unknown," preventing misclassifications.
* **True On-Device Deployment & Power Profiling:** The next logical step is to port the model to a mobile-native format (e.g., TensorFlow Lite or Core ML) and deploy it on an actual Android or iOS device. This would allow for crucial **power profiling** to measure the real-world impact on battery life—a key metric for any on-device AI.
* **Actionable Policy Integration:** The current system is a classifier. The final step to complete the vision is to integrate its output with the device's networking stack. For example, when "Reel Traffic" is detected, the device could use a 5G **URSP (UE Route Selection Policy)** to request a high-throughput, low-latency network slice from the carrier, thus closing the loop from detection to action.
* **Cross-Platform Generalization:** To create a truly universal model, the dataset would be expanded to include traffic from more SNS apps (TikTok, Facebook), different network types (5G, congested public Wi-Fi), and a wider range of devices.

---

## 8. Installation and Usage Guide

### Important Note on Live Demonstration

This model was specifically trained and optimized on data captured from the native **YouTube** application. The statistical "fingerprint" of traffic can vary significantly between different applications (e.g., YouTube vs. Instagram) and between a native app and a web browser.

For the most accurate and representative results during the live demonstration, please ensure you are generating traffic from the **YouTube application** while the dashboard is running.

### Installation
1.  Clone the repository and navigate into the project folder.
2.  Create and activate a Python virtual environment.
3.  Install all required libraries: `pip install -r requirements.txt`

### Usage Guide
All scripts should be run from the project's **root directory**.

* **To create the augmented dataset:**
    ```bash
    python src/augment_data.py
    ```
* **To train the final robust model:**
    ```bash
    python src/train_model.py
    ```
* **To run the live dashboard:**
    ```bash
    streamlit run src/dasboard.py
    ```
---

## 9. Ethical Considerations & Scalability

* **Ethical:** Our solution is **privacy-preserving by design**. It operates on encrypted traffic metadata and never inspects user data content, avoiding significant ethical and legal concerns.
* **Scalability:** The lightweight nature of the LightGBM model makes it highly scalable for deployment on millions of individual user devices (edge computing) without significant battery or performance impact.