import pdb
import pandas as pd



def create_balanced_dataset(df):

    # Counts the instances of “spam”
    num_spam = df[df["Label"] == "spam"].shape[0]

    # Randomly samples “ham” instances to match the number of “spam” instances
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # Combines ham subset with “spam”
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df



def main():
    data_file_path = "./ch06/data/sms_spam_collection/SMSSpamCollection.tsv"
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    # print(df)

    # class label distribution, label “ham” (i.e., not spam)
    print(df["Label"].value_counts())
    balanced_df = create_balanced_dataset(df)
    print(balanced_df["Label"].value_counts())

    # we convert the “string” class labels "ham" and "spam" into integer class labels 0 and 1, respectively:
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    



if __name__=="__main__":
    main()