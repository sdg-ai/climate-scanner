# -*- coding: utf-8 -*-
import os
import yaml
from climate_scanner.trends_innovation_classifier.data_utils import doc_to_sentence, doc_to_multisentence, data_processing, load_params
import glob
import os
import spacy
from operator import itemgetter

_ROOT = os.path.abspath(os.path.dirname(__file__))

def get_full_path(path):
    return os.path.join(_ROOT, path)

# loading config params
with open(os.path.join(_ROOT, 'data', "config.yml")) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

params = load_params()

class TrendsInnovationClassifier:
    """A class to define functions to train/test/predict and evalute the classifier.
    """

    def __init__(self, debug=False, mode="multi"):
        self.debug = debug
        self.mode = mode.lower()
        self.categories, self.model_names = self._list_all_classifiers()
        self.models = self.load_models(self.mode)

    def _list_all_classifiers(self):
        """
        A function which lists the available classifiers.
        :return: list of categories of trends and innovations for which there are classifiers.
        """
        categories = []
        model_names = []
        inputfilepath = get_full_path(params['data']['path_to_prodigy_models'])
        print(inputfilepath)
        # filenames = glob.glob(inputfilepath)
        # my_list = os.listdir('My_directory')
        filenames = [x[0] for x in os.walk(inputfilepath)]
        print(filenames)

        for fn in filenames:
            head, tail = os.path.split(fn)
            print(head, tail)
            categories.append(tail)
        print(categories, model_names)
        print(len(categories), len(model_names))
        filtered_filenames = []
        filtered_categories = []
        for filepath, category in zip(filenames, categories):
            if '_model' in category:
                filtered_filenames.append(filepath)
                filtered_categories.append(category)

        print(filtered_filenames, filtered_categories)

        return filtered_categories, filtered_filenames

    def load_all_classifiers(self, model_names):
        """
        A function which loads the available classifiers.
        """

        all_classifiers = []

        for model in model_names:
            print(model)
            try:
                model_best = os.path.join(model,'model-best')
                all_classifiers.append(spacy.load(model_best))
            except Exception as e:
                print(e)

        return all_classifiers

    def load_models(self, mode):
        """
        A function which loads the model(s).
        :inputs:
            mode (str): whether to load single or multi label models
        """

        classifiers = []

        # If 'single' mode, load single label classifiers.
        if mode == "single":
            for model in self.model_names:
                print(model)
                try:
                    model_best = os.path.join(model,'model-best')
                    classifiers.append(spacy.load(model_best))
                except Exception as e:
                    print(e)

        # If 'multi' mode, load multi label classifier.
        elif mode == "multi":
            try:
                model_best = os.path.join(get_full_path(params['data']['path_to_prodigy_models']), 'balanced_multilabel_classifier', 'model-best')
                classifiers = spacy.load(model_best)
            except Exception as e:
                print(e)

        else:
            print("Invalid mode entered. Set mode to either 'single' or 'multi' label.")

        return classifiers

    def predict(self, text):
        """
        A function which predicts the output from all available classifiers.
        :inputs:
            text: text to predict.
        :return:
            most_likely_trends: list of most likely trends and innovations.
        """
        all_labels, all_probs, most_likely_trends = [], [], []
        threshold = params['trends_demo']['threshold']
        count = params['trends_demo']['count']
        if self.debug:
            print(count, threshold)
            print(len(self.models))

        # If 'single' mode, iterate through the single label classifiers.
        if self.mode == "single":
            for nlp in self.models:
                doc = nlp(text)
                prediction = doc.cats
                for item in prediction:
                    predict_label = item
                    predict_prob = prediction[predict_label]
                if self.debug:
                    print("\nmodel: {} \t predict_label {} \t predict_prob {}: ".format(self.categories[self.models.index(nlp)],
                                                                                        predict_label, predict_prob))
                all_labels.append(predict_label)
                all_probs.append(predict_prob)

        # If 'multi' mode, make prediction using the multi label classifier.
        elif self.mode == "multi":
            nlp = self.models
            doc = nlp(text)
            prediction = doc.cats
            print("Prediction: ", prediction)
            for item in prediction:
                all_labels.append(item)
                all_probs.append(prediction[item])
            print("\nmodel: {} \t predict_label {} \t predict_prob {}: ".format(
                self.models, all_labels, all_probs))

        else:
            print("Invalid mode entered. Set mode to either 'single' or 'multi' label.")
            
        most_likely_trends = [[all_labels[i], all_probs[i]] for i, v in enumerate(all_probs) if v > threshold]
        most_likely_trends = sorted(most_likely_trends, key=itemgetter(1), reverse=True)
        if self.debug:
            print("most likely trends contains: ", most_likely_trends)
        try:
            most_likely_trends = [most_likely_trends[i] for i in range(count)]
        except:
            print("{} trends match the requested threshold.".format(len(most_likely_trends)))

        return most_likely_trends

    def scan_predict(self, text):
        """ Function to scan over three sentence blocks and make predictions
        :arg text (str) - input text string
        @:return array([{"text": <text-snippet>,
                        "indices": [(<start>, <end>)],
                         "prediction": <tag str>}]"""
        enriched_sentence_objects = []
        tags = set()
        sentencs_objects = doc_to_multisentence(text, 3)
        # Predict over sentence blocks
        for sentence_block in sentencs_objects:
            text_block = ' '.join(sentence_block['text'])
            predictions = self.predict(text_block)

            sentence_block['predictions'] = {}
            for item in predictions:
                sentence_block['predictions'][item[0]] = item[1]

            for tag, confidence in predictions:
                tags.add(tag)
            if predictions:
                enriched_sentence_objects.append(sentence_block)

        # Condense predictions
        predictions = []
        block_count = len(enriched_sentence_objects)
        for tag in tags:
            current_index_set = []
            current_sentence_set = []
            current_end_offset = None
            i = 0
            while i < block_count:
                if tag in enriched_sentence_objects[i]['predictions']:
                    if not current_end_offset:
                        current_end_offset = i
                        for item in enriched_sentence_objects[i]['string_indices']:
                            if item not in current_index_set:
                                current_index_set.append(item)
                        for item in enriched_sentence_objects[i]['text']:
                            if item not in enriched_sentence_objects:
                                current_sentence_set.append(item)


                    elif i - current_end_offset < 3:
                        current_end_offset = i
                        for item in enriched_sentence_objects[i]['string_indices']:
                            if item not in current_index_set:
                                current_index_set.append(item)
                        for item in enriched_sentence_objects[i]['text']:
                            if item not in enriched_sentence_objects:
                                current_sentence_set.append(item)

                    else:
                        # Format prediction object
                        prediction_obj = {'string_indices': [list(current_index_set)[0][0],
                                                              list(current_index_set)[-1][1]],
                                          'text': ' '.join(current_sentence_set),
                                          'prediction': tag}
                        if self.debug:
                            prediction_obj['text'] = ' '.join(current_sentence_set)
                        predictions.append(prediction_obj)

                        # Reset variables
                        current_end_offset = i
                        current_index_set = []
                        current_sentence_set = []

                        for item in enriched_sentence_objects[i]['string_indices']:
                            if item not in current_index_set:
                                current_index_set.append(item)
                        for item in enriched_sentence_objects[i]['text']:
                            if item not in enriched_sentence_objects:
                                current_sentence_set.append(item)

                i += 1
            if current_index_set:
                # Format prediction object
                prediction_obj = {'string_indices': [list(current_index_set)[0][0],
                                                      list(current_index_set)[-1][1]],
                                  'text': ' '.join(current_sentence_set),
                                  'prediction': tag}
                if self.debug:
                    prediction_obj['text'] = ' '.join(current_sentence_set)
                predictions.append(prediction_obj)

        return predictions



    def demo_return(self, text):
        """
        A demo function to predict the most likely trends and innovations.
        :inputs: text to classify.
        :return: dictionary containing the classified trends and innovations.
        """

        most_likely_trends = self.predict(text)


        output = {"string_prediction": [most_likely_trends[i][0] for i in range(len(most_likely_trends))],
                  "string_prob": [most_likely_trends[i][1] for i in range(len(most_likely_trends))]}

        return output

    def train(self):
        """
        A function to train the classifier.
        :return:
        """
        pass

    def test(self):
        """
        A function to test the classifier.
        :return:
        """
        pass


    def eval(self):
        """
        A function to evaluate the classifier.
        :return:
        """
        pass

class Doc2Trends:

    def __init__(self):
        self.tic = TrendsInnovationClassifier()


    def coordinator_pipeline(self,input_dict=None,threshold=None):
        id_dict = dict()
        id_dict["ID"] = input_dict["ID"]
        if input_dict["climate_scanner_prob"] >= threshold:
            text = input_dict["doc"]
            split_sentences = doc_to_sentence(text)
            all_predictions = []
            for dic in split_sentences:
                sentence = data_processing(dic["text"])
                output = self.tic.predict(sentence)
                dic.update(output)
                all_predictions.append(dic)
        else:
            all_predictions = []
        return [id_dict] + all_predictions


def main():
    """
    pipeline to read data/process data /train/ test /evaluate
    :return:
    """
    x = TrendsInnovationClassifier()
    text = "Lithium ion battery life has greatly improved electric transport. " \
           "Electric motors can now run for hundreds of kilometers without needing to refuel. " \
           "Car manufacturing is improving year on year"

    text = "Devolved and local government play an essential role in meeting national net zero ambitions. Across the UK many places have already made great strides towards our net zero future, having set their own targets and strategies for meeting local net zero goals. Taking a place-based approach to net zero is also vital to ensuring that the opportunities from the transition support the government’s levelling up agenda. 2. The combination of devolved, local, and regional authorities’ legal powers, assets, access to targeted funding, local knowledge, and relationships with stakeholders enables them to drive local progress towards net zero. Not only does local government drive action directly, but it also plays a key role in communicating with, and inspiring action by, local businesses, communities, and civil society. Of all UK emissions, 82% are within the scope of influence of local authorities.43 3. Local leaders are well placed to engage with all parts of their communities and to understand local policy, political, social, and economic nuances relevant to climate action. The government currently works with the Core Cities Group, for instance, which undertakes a range of activities to promote climate change adaptation, raise awareness and foster leadership in cities. Local government decides how best to serve communities and is best placed to integrate activity on the ground so that action on climate change also delivers wider benefits – for fuel poor households, for the local economy, for the environment and biodiversity, as well as the provision of green jobs and skills. 4. Despite the excellent work already underway, we understand that there remain significant barriers to maximising placebased delivery on net zero. We know that some places are moving faster than others and that places and communities will face different challenges when meeting net zero commitments and adapting to climate change. 5. There are significant regional variations in the level of emissions (see Figure 29 below) and some of the hardest hit local economies that face multiple development and growth challenges are proportionally home to a greater number of lower skilled workers. Many of these areas are also where high-carbon industries are locate"
    output = x.scan_predict(text)
    # print(output)


if __name__ == "__main__":
    main()
