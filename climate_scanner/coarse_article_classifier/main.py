from climate_scanner.coarse_article_classifier.coarse_classifier import classifier

classifier_obj = classifier()

path = ""
data = classifier_obj.load_data(path)
transformed_data = classifier_obj.pre_processing(data)
model = classifier_obj.fit(transformed_data)
predictions = classifier_obj.predict(model,transformed_data)

classifier_obj.evaluate(predictions)
