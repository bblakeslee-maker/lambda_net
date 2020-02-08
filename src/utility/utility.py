def gramMatrix(activationMap):
    (batchSize, channels, height, width) = activationMap.size()
    features = activationMap.view(batchSize, channels, width * height)
    transposeFeatures = features.transpose(1, 2)
    gram = features.bmm(transposeFeatures) / (channels * height * width)
    return gram


def normalizeBatch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.4931, 0.4732, 0.4457]).view(-1, 1, 1)
    stdDev = batch.new_tensor([0.2264, 0.2212, 0.2295]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / stdDev
