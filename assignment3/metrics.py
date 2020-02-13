def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
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
    
    
    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    correct = 0
    for i in range(len(prediction)):
        if prediction[i]==ground_truth[i]:
            correct += 1
    return correct/len(prediction)