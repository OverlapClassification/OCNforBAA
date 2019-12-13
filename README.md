## Overlap Classification Network for Skeletal Bone Age Assessment.

This work not designed a new network architecture, but proposaled a new classification method. The most important contribution of this work is a label pre-processing concept and a new loss function for BAA. This work also has implications for solving other classification or regression problems.

The work used a calssic CNN inception-v3 and based on a public database of RSNA. These codes only offer one implemention way to achieve overlap classification, and its not the unique way to achieve overlap classification.

### Abstract
The bone development is a continuous process, however, discrete labels are usually used to represent bone ages. This inevitably causes a semantic gap between actual situation and label representation scope. In this paper, we present a novel method named as overlap classification network to narrow the semantic gap in bone age assessment. In the proposed network, discrete bone age labels (such as 0-228 month) are considered as a sequence that is used to generate a series of subsequences. Then the proposed network utilizes the over- lapping information between adjacent subsequences and out- put several bone age ranges at the same time for one case. The overlapping part of these age ranges is considered as the final predicted bone age. The proposed method without any pre- processing can achieve a much smaller mean absolute error compared with state-of-the-art methods on a public dataset.
### Overview
#### Here gives an overview when we use 2 bone age range labels to replace an original bone age label. (It is k = 2)
![overview](Img/overview.png)
#### In Our experiment we adopt k = 3 (use 3 bone age range labels to replace an original bone age label)
### Experiment samples
#### Some Output examples are shown below
<img src="Img/Output.png"  height="500" >

#### Some results comparison between this work and traditional method are shown bellow.
<img src="Img/compare.png"  height="500" >
