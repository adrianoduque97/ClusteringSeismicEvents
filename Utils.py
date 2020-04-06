def toPrediction(list_clusters:list):
    prediction = {}

    for row in range(len(list_clusters)):
        for index in list_clusters[row]:
            prediction[index]=row
    prediction_list = []

    for i in range(len(prediction)):
        prediction_list.append(prediction[i])
    return prediction_list