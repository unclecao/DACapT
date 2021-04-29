# DACapT, Feature Aggregation Networks based on Dual Attention Capsules for Visual Object Tracking
A new tracker based on Capsule networks, which is accepted by IEEE Transactions on Circuits and Systems for Video Technology.

# Requirements
- PyTorch 0.4.1
- Python 2.7
- opencv

# Abstract
Tracking-by-detection algorithms have considerably enhanced tracking performance with the introduction of recent convolutional neural networks (CNNs). However, most trackers directly exploit standard scalar-output CNN features, which may not capture enough feature encoding information, instead of aggregated CNN features of vector-output form. In this paper, we propose an end-to-end feature aggregation capsule framework. First, based on the existing CNN network, we aggregate a certain number of similar position-aware CNN features into a capsule to model the feature similarity. The acquired vector-level feature capsules (rather than previous scalar-level pointwise features) are utilized for differentiation learning. We then propose a group attention module to better model the entity representation between different capsule groups thus optimizes total discriminative capability. Third, to reduce the prediction interference resulted by the side effect of dimension rising within capsules, we propose a penalty attention module. Such strategy could dynamically adjust values of neurons by estimating whether they are beneficial or harmful to tracking. Experimental results on five representative benchmarks (UAVDT, DTB70, UAV123, VOT2016 and VOT2018) demonstrate the excellent tracking performance of our dual attention based capsule tracker (DACapT). Specially, it exceeds the previous top tracker by 4.6% /1.9% in precision/success evaluations on UAVDT.

# Publication and citation
Cao Y, Ji H, Zhang W, Shirani S. Feature Aggregation Networks based on Dual Attention Capsules for Visual Object Tracking[J].
IEEE Transactions on Circuits and Systems for Video Technology, 2021.

# To be noted
Our work is the extension of tracking-by-detection method MDNet (https://github.com/hyeonseobnam/py-MDNet/). 
And thanks for their open source codes, we maintain the similar tracking procedures.

For the capsule construction, we are inspired by the work of https://github.com/cedrickchee/capsule-net-pytorch.
