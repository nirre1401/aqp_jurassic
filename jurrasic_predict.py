import pandas as pd
import glob
import ai21
ai21.api_key = 'bH13kzCYtAxPbw7iVE5zX0Och2GxI3bl'
from pathlib import Path
import numpy as np
import math


# J2 Mid
include_list = ['elections']

def evaluate(path):

    files = "{}/*.csv".format(path)
    for fname in glob.glob(files):

        df_queries = pd.read_csv(fname)
        df_queries["predicted_result"] = 0
        custom_model = Path(fname.split("_")[3]).stem
        if custom_model not in include_list:
            continue
        print("Start processing ",fname)
        for index, row in df_queries.iterrows():
            print("Trying to predict ", row.prompt)

            if row.predicted_result==0:
                for i in range(10):
                    try:
                        response_mid = ai21.Completion.execute(
                            model="j2-light",
                            custom_model=custom_model,
                            prompt="what is the result of this query: {}".format(row.prompt),
                            numResults=1,
                            epoch=5,
                            maxTokens=200,
                            temperature=0.7,
                            topKReturn=0
                        )
                        predicted_result = float(response_mid.values["completions"][0]["data"]["text"])
                        df_queries.at[index, "predicted_result"] = predicted_result
                        print("predicted result", predicted_result)
                        break
                    except:
                        print("Query failed, retrying")
                        if df_queries.at[index, "predicted_result"] == 0:
                            print("too many retries, moving to the next record")
            else:
                break

        print("Done predicting", custom_model)
        print("calc NRMSE")
        y_actual = df_queries.completion
        y_predicted = df_queries.predicted_result

        MSE = np.square(np.subtract(y_actual, y_predicted)).mean()
        NRMSE = math.sqrt(MSE)/(np.max(y_actual)-np.min(y_actual))
        df_queries['NRMSE'] = NRMSE
        df_queries.to_csv("./results/{}_results.csv".format(custom_model,custom_model))

    # Press the green button in the gutter to run the script.
if __name__ == '__main__':
    evaluate("./validation_queries")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
