# Imports
from sklearn.linear_model import LogisticRegression

class Classifier():
    def __init__(self, )
        log.info("Initializing classifier object")

    def load_data(self, path):
        """
        function that will get the necessary data and return a dataframe
        """
        data = {
                "ID" : 1545,
                "title" : "The Shattering Truth of 3D-Printed Clothing",
                "doc" : "Three-dimensional printing has changed the way we make everything from prosthetic limbs to aircraft parts and even homes.Now it may be poised to upend the apparel industry as well.Fashion designers have already unveiled shoes and clothing made via 3D printing, in which plastic material is deposited layer upon layer to create a three-dimensional structure. In one recent example, Dutch designer Iris van Herpen showed off a 3D-printed dress at last month's Paris Fashion Week.3D printing won't replace weaving, knitting, and other conventional means of apparel manufacturing anytime soon, given its high cost and how difficult it is to make durable 3D-printed 'fabrics' that are soft to the touch and which drape like traditional fabrics.But some experts foresee a day when we could print out customized garments right in the store, or maybe even from 3D printers in our own homes. 'Imagine having a garment fit exactly to your size and preferences,' Melissa Dawson, an assistant professor of industrial design at the Rochester Institute of Technology and a 3D printing expert, told NBC News MACH in an email. 'You could also customize your color and pattern choices… maybe even trims and finishes.'Digital clothing on demandDanit Peleg is on the cutting edge of 3D-printed apparel. The Tel Aviv-based designer — known for creating a dress worn by American snowboarder Amy Purdy at the opening ceremony of the 2016 Paralympic Games in Rio de Janeiro — says 3D-printed clothing is an all-but-inevitable part of the ongoing digital revolution.'We used to buy CDs, and we had to go to the physical stores to get music and now we can just download it everywhere,' she says. 'I believe that the same thing will happen with fashion eventually — clothes will become more and more digital.'Peleg's first 3D-printed dresses were made of scratchy plastic that bothered the models who wore them. But she found she could use FilaFlex, a flexible, rubbery material that she says 'fits the curve of the body really nicely.' Now she can print an entire outfit, including accessories like shoes and sunglasses.She's also printed dresses and skirts that she wears to conferences and a bomber jacket that can be personalized and ordered online for $1,500. Eventually, she hopes, people will make her garments in their own homes.Big pluses, but a few snagsIf convenience and customization are potential advantages of 3D-printed apparel, so is recyclability.Conventional clothing can be broken down and turned into new fibers, but only about 0.1 percent of the textiles collected by charities and take-back programs actually gets recycled, Newsweek has reported. And the fabric scraps left over from manufacturing a new shirt or dress via conventional means are typically thrown out.In contrast, 3D-printed clothes can simply be dumped into blenderlike machines that turn the plastics into powder that can then be used to print out something new. And since 3D printing easily allows for custom sizing, the process is inherently frugal with materials.But there are plenty of challenges that must be overcome before 3D-printed apparel goes mainstream.One is cost. Even the smallest home 3D printers run several hundred dollars. A printer capable of printing human-sized apparel is beyond the reach of individual consumers. And it takes far longer to print an article of clothing than to produce a similar article via weaving or knitting. Peleg's jacket, for example, takes about 100 hours to print.Then there's the matter of comfort. Since it's made of plastic, 3D-printed fabric tends to be stiffer and less comfortable than traditional fabric (Peleg's jacket has a fabric lining). So while a 3D-printed dress might be fine on the runway or red carpet, it 'doesn't really make sense yet' for everyday use, says Elizabeth Esponnette, co-founder of the San Francisco-based on-demand clothing startup Unspun.Moving forwardEsponnette thinks it might take a couple of decades before 3D-printed clothing is ready for prime time. For now, she says, it remains the province of designers rather than major clothing manufacturers.But recent innovations are advancing the nascent technology. Some designers are creating softer, more flexible fabrics by linking together many small pieces of 3D-printed material like chainmail, Dawson says. She's also developing a technique to knit together 3D-printed filaments and has used them to create a dress that she says stretches and drapes like one knit from regular fabric.Meanwhile, several companies are developing 3D-printed shoes, including Nike and New Balance. Adidas began selling shoes with 3D-printed soles in January; following the initial limited release, it aims to mass-produce 100,000 pairs by year-end. In the future, the company hopes to offer footwear customized for its customers' feet.Dawson says 3D-printed shoes could become common in the next 10 years. Shoes could also be more accessible for people to print at home than clothing since they're smaller and thus less expensive. This would be especially true for children's shoes, she says: 'Can you imagine printing out a new pair of shoes for your child over breakfast?"
            }

        return data

    def pre_processing(self, data):
        """
        function to clean the data, numerify the data without losing too much information (feature engineering)
        """  
        final__data = {
                "ID" : 1545,
                "title" : "The Shattering Truth of 3D-Printed Clothing",
                "cleaned_doc" : "  "
            }

        return final__data

    def fit(self, final__data):
        """
        1. split data to training and test set
        2. fit the model on training data
        3. parameter tuning
        """
        x_train, y_train, x_test, y_test = # training and test set

        # create an object for the model being used. For the sake of understanding, I'm creating an object for Logistic regression here.

        self.model = LogisticRegression()
        self.model.fit(x_train, y_train)

        self.x_test = x_test
        self.y_test = y_test
    
    def predict(self, test__data):
        """
        predict classes (relevant/not relevant) for the test data
        """
        y_pred = self.model.predict(self.x_test)

        predictions = {
                "ID" : 1545,
                "title" : "The Shattering Truth of 3D-Printed Clothing",
                "climate_scanner" : True,
                "climate_scanner_prob" : 0.9
            }

        return predictions
    
    def evaluate(self, predictions):
        """
        evaluate the predictions
        """
