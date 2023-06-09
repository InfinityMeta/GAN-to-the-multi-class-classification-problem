GANs to the multi-label classification problem 

Deep learning models need more training samples for higher performance.
One solution is to use data augmentation providing new samples based on
instances from dataset. Application of generative adversarial networks (GAN)
is one of the ways to implement this technique. Use of image embeddings,
which are dense vector representations of images, is an alternative option to
get better performance. In this paper, combination of GAN augmentation
and application of embeddings in the form of concatenated probability
vectors is introduced. Three-stage pipeline for the task of fine art paintings
classification by an artistic style is demonstrated. On the first stage paintings
undergoes the procedure of division by five equal parts, which are along
with entire images separately passed to the pretrained convolutional neural
network InceptionV3 serving as a probability vectors extractor. On the
second stage extracted probability vectors are concatenated to a single
embeddings forming new dataset and Wasserstein GAN in regular and
conditional variants is used for augmentation of embeddings. The third
stage is represented by a shallow neural networks assigning final labels to
samples. The experiments with filtered and not filtered synthetic samples
were conducted. Total classification accuracy and mean accuracy for three
styles with the lowest percentage of correct predictions are used for evaluating
the pipeline‘s performance. The best improvements are obtained for the
pipeline containing regular WGAN without filtering.

