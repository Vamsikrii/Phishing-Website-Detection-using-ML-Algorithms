# Phishing-Website-Detection-using-ML-Algorithms
Developed and tested 6 different machine learning algorithms to identify most effective method for detecting Phishing websites. Compared performance metrics including Accuracy, Precision, F1 score, Log Loss to evaluate the strengths and weaknesses of each algorithm.
1.Introduction
1.1 What is Phishing? 
Phishing is a widespread technique used by cybercriminals to obtain personal data from 
unsuspecting individuals. To mitigate the damage caused by phishing attacks, early detection 
is crucial. Nearly all phishing attacks involve phishing URLs, which are links to websites 
designed to mimic legitimate sites but are malicious. These malicious sites are created by 
cybercriminals to steal personal information such as login credentials, credit card numbers, 
and other sensitive data. Detecting phishing URLs is critical due to the significant threat they 
pose to both individuals and organizations. The importance of identifying these malicious links 
can be summarized as follows: 
1. Protection against Identity Theft: Phishing URLs are crafted to deceive individuals into 
revealing their login credentials, bank account details, and other personal information. Early 
detection helps protect against identity theft. 
2. Prevention of Financial Loss: Phishing attacks can lead to substantial financial losses. By 
identifying and blocking phishing URLs, organizations can prevent cybercriminals from 
stealing money and sensitive data. 
3. Protection against Malware: Phishing URLs often contain links to malicious software. 
Detecting and blocking these URLs can prevent malware infections and data breaches. 
4. Maintaining Trust: Organizations affected by phishing attacks can lose the trust of their 
customers, clients, and partners. Effective detection and prevention help maintain an 
organization's reputation and avoid negative publicity. 
In summary, detecting phishing URLs is essential for protecting against identity theft, financial 
loss, malware infections, and maintaining trust. Vigilance in identifying and reporting phishing 
URLs is crucial for staying safe in the digital world. 
1.2 Background 
Phishing URLs and Machine Learning Phishing URLs are malicious links distributed by 
attackers to trick users into revealing sensitive information such as passwords, credit card 
numbers, and other personal data. Machine learning artificial intelligence technique involves 
training computer algorithms on large datasets. In the context of phishing URL detection, 
machine learning can identify suspicious patterns in URLs that may indicate phishing 
attempts. 
Machine learning models used for phishing URL detection analyze large datasets, considering 
various features such as the URL structure, the appearance of the web page, and contextual 
information. These models are trained on real examples of both phishing and non-phishing 
sites, enabling them to identify suspicious links based on the learned patterns. 
7 
1.3 Phishing Detection Methods 
To enhance phishing detection, various methods are employed, including: 
1. Feature-Based Analysis: Extracting and analyzing features of URLs, such as length, 
presence of special characters, and use of HTTPS. 
2. Content-Based Analysis: Examining the content of web pages, including text, images, and 
layout, to identify discrepancies that indicate phishing. 
3. Blacklisting: Maintaining a list of known phishing URLs and blocking access to them. 
4. Heuristic-Based Detection: Using rules and algorithms to identify suspicious patterns in 
URLs and web content. 
5. Machine Learning Algorithms: Implementing advanced algorithms such as Decision Tree, 
Random Forest, Support Vector Machine (SVM), XGBoost, Backpropagation Neural Network 
(NN), and Convolutional Neural Network (CNN) to detect phishing URLs. 
6. Hybrid Methods: Combining multiple detection techniques to improve accuracy and reduce 
false positives and false negatives. 
1.4 Phishing Evaluation Methods 
In this project, different machine learning algorithms are evaluated using the following 
methods: 
1. Accuracy Rate: The percentage of correctly classified phishing and legitimate URLs. 
2. False Positive Rate: The percentage of legitimate URLs incorrectly classified as phishing. 
3. False Negative Rate: The percentage of phishing URLs incorrectly classified as legitimate. 
4. Precision: The ratio of true positive predictions to the total predicted positives. 
5. Recall: The ratio of true positive predictions to the total actual positives. 
6. F1 Score: The harmonic mean of precision and recall, providing a balance between the two. 
By leveraging these methods, especially the power of machine learning, it is possible to 
effectively detect phishing URLs and protect users from cyberattacks related to phishing. 
8 
1.5 The popularity of phishing attacks 
 The popularity of phishing attacks Phishing attacks are as old as the Internet itself. The data 
available today show that as the Internet evolves, phishers also evolve their tactics and 
construct more elaborate attacks. This is backed by the following statistics, which capture the 
trends in phishing from the perspective of the renowned organization: The APWG 
The APWG 
Anti-Phishing Working Group (APWG) is an international coalition of more than 2200 
members, which unifies the global response to cybercrime across industrial, governmental, 
law enforcement, and non-governmental organization sectors. Alongside other activities, their 
research efforts produce a quarterly Phishing Activity Trend Reports, which analyze phishing 
attacks reported by the members of the APWG. Figure 1.5 shows yearly figures of unique 
phishing websites detected by the APWG. 
Figure 1.5: Yearly number of unique phishing sites detected by APWG in 2021-2023 
The APWG observed almost five million phishing attacks over the course of 2023, which was 
a record year. 
9 
1.6 Introduction to Machine Learning Techniques 
1. Decision Tree: Decision Trees are a straightforward machine learning algorithm used 
for classification and regression. They work by splitting data into branches based on 
feature values, creating a tree-like structure. Each node represents a decision point, 
making the model easy to interpret. However, decision trees can overfit, particularly 
with complex data, unless pruned. They are often used as a baseline model due to their 
simplicity and ease of implementation. 
2. Random Forest: Random Forest is an ensemble method that builds multiple decision 
trees and combines their predictions to improve accuracy and control overfitting. Each 
tree is built from a random subset of data and features, reducing variance. This method 
is robust and effective for complex datasets, handling large amounts of data efficiently 
and providing insights into feature importance, making it widely used in various 
domains. 
3. XGBoost: XGBoost, or eXtreme Gradient Boosting, is an enhanced gradient boosting 
algorithm known for its speed and performance. It uses regularization techniques to 
prevent overfitting and improve generalization. With parallel computation capabilities, 
XGBoost efficiently handles large datasets. Its high predictive accuracy and versatility 
have made it a top choice in competitive machine learning tasks. 
4. Support Vector Machine (SVM): Support Vector Machines are powerful algorithms 
for classification and regression, effective in high-dimensional spaces. They find a 
hyperplane that best separates data into classes, making them robust to outliers and 
overfitting. SVMs can handle nonlinear problems using kernel tricks, offering 
flexibility for complex datasets. They are popular for tasks like text classification and 
image recognition. 
5. Neural Network (Backpropagation): Neural Networks are models inspired by the 
human brain's architecture, consisting of interconnected layers that learn complex 
patterns. Backpropagation is a technique to adjust weights through gradient descent, 
minimizing prediction errors. This enables neural networks to model intricate data 
relationships, making them suitable for tasks such as image and speech recognition. 
6. Convolutional Neural Network (CNN): Convolutional Neural Networks are 
specialized for processing structured grid data like images. They capture spatial 
hierarchies through convolutional layers that learn local patterns. CNNs consist of 
convolutional, pooling, and fully connected layers, making them robust for image 
classification, object detection, and vision-related tasks. Their ability to automatically 
learn spatial hierarchies has driven advances in computer vision. 
10 
2. Literature Review
The literature review section provides an in-depth analysis of existing phishing detection 
techniques and explores the application of machine learning in this domain. By examining 
previous research, this section aims to highlight the evolution of phishing detection methods 
and the advantages of using machine learning. 
2.1 Overview of Existing Phishing Detection Techniques 
Phishing detection has been a significant focus within cybersecurity due to the increasing 
prevalence of phishing attacks. Various techniques have been developed over the years to 
detect and mitigate these attacks. The existing phishing detection techniques can be broadly 
classified into the following categories: 
1. Blacklisting: 
 - Description: Blacklisting involves maintaining a list of known phishing URLs and 
blocking access to these URLs. 
 - Advantages: Simple to implement and effective against known threats. 
 - Disadvantages: Ineffective against new, unknown phishing URLs. The blacklist needs 
constant updating, which can be resource-intensive. 
2. Heuristic-Based Detection: 
 - Description: Heuristic-based detection uses a set of predefined rules to identify phishing 
URLs. These rules are based on common characteristics of phishing websites, such as the 
use of certain keywords or suspicious URL patterns. 
 - Advantages: Can detect unknown phishing URLs if they match the predefined rules. 
 - Disadvantages: Limited by the quality and comprehensiveness of the heuristics. High false 
positive rate due to the generality of the rules. 
3. Content-Based Analysis: 
 - Description: Content-based analysis examines the content of web pages, including text, 
images, and layout, to identify discrepancies that indicate phishing. This method may 
involve comparing the content to known legitimate websites. 
11 
 - Advantages: Can detect phishing sites that are visually similar to legitimate sites. 
 - Disadvantages: Computationally intensive and can be evaded by attackers using dynamic 
content generation. 
Machine Learning-Based Techniques 
 Machine learning techniques use statistical models and algorithms to classify URLs as 
phishing or legitimate. These models are trained on features extracted from URLs, 
such as lexical features, host-based features, and content-based features. 
 
2.2 Comparison of Traditional and Machine Learning-Based Methods 
Traditional phishing detection methods, such as blacklisting and heuristic-based detection, 
have been effective to some extent. However, they suffer from several limitations, including 
the inability to detect new phishing URLs and high false positive rates. Machine learningbased methods offer several advantages over traditional approaches: 
1. Adaptability: 
 - Machine learning models can adapt to new phishing techniques by learning from updated 
datasets. Traditional methods rely on predefined rules that may become obsolete. 
2. Accuracy: 
 - Machine learning algorithms, particularly those using advanced techniques like deep 
learning, can achieve higher accuracy in detecting phishing URLs. Traditional methods may 
struggle with accuracy due to their simplistic nature. 
3. Scalability: Machine learning models can handle large volumes of data and detect 
phishing URLs at scale. Traditional methods, especially manual blacklisting, are not 
scalable. 
2.3 Machine Learning-Based Techniques
Description:
 Machine learning techniques use statistical models and algorithms to classify URLs as 
phishing or legitimate. These models are trained on features extracted from URLs, 
such as lexical features, host-based features, and content-based features. 
Types of Machine Learning Models:
12 
 Supervised Learning: Models are trained using labelled data (URLs marked as 
phishing or legitimate). Common algorithms include Decision Trees, Random Forest, 
Support Vector Machines (SVM), and Neural Networks. 
 Unsupervised Learning: Models identify patterns and anomalies without labeled 
data. Techniques include clustering and anomaly detection. 
Strengths:
 High accuracy and adaptability to new phishing tactics. 
 Can handle large datasets and complex feature sets. 
Limitations:
 Requires a large amount of labelled data for training. 
 Model performance depends on the quality of features extracted from URLs. 
 May require significant computational resources for training and deployment. 
2.4 Deep Learning-Based Techniques
Description:
 Deep learning techniques leverage neural networks with multiple layers to 
automatically learn features and patterns from raw data. These models can analyze the 
entire URL structure, including its components and context. 
Types of Deep Learning Models:
 Convolutional Neural Networks (CNN): Effective in capturing spatial hierarchies in 
data, making them suitable for analyzing the structure of URLs. 
 Neural Network (Backpropagation): Backpropagation is a key algorithm for 
training neural networks, particularly those with multiple layers (deep networks). It 
involves a forward pass, where the input data is passed through the network to 
generate predictions, and a backward pass, where the error between predictions and 
actual values is propagated back through the network. 

