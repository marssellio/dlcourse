def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    success=0
    fail = 0
    for i in range(len(prediction)):
        if ground_truth[i]:
            if prediction[i]:
                success +=1
            else:
                fail +=1
    precision = (success/(success+fail))
    recall = 0
    tp=0
    fp=0
    for i in range(len(prediction)):
        if ground_truth[i]:
            if prediction[i]:
                tp +=1
            else:
                fp+=1
    tn=0
    fn=0
    for i in range(len(prediction)):
        if ground_truth[i] == False:
            if prediction[i] == False:
                tn +=1
            else:
                fn +=1
    recall = tp/(tp+fn)
    accuracy = 0
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    f1 = 2*(precision*recall)/(precision+recall)
    
    
    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    correct = 0
    for i in range(len(prediction)):
        if prediction[i]==ground_truth[i]:
            correct += 1
    return correct/len(prediction)
