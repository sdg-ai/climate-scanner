from climate_scanner.coarse_article_classifier.coarse_classifier import classifier


path = ""
data = classifier.load_data(path)
transformed_data = classifier.pre_processing(data)
model = classifier.fit(transformed_data)
predictions = classifier.predict(model,transformed_data)

classifier.evaluate(predictions)
