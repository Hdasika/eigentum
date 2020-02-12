import sys

from pyspark.ml.feature import RegexTokenizer
from pyspark.sql import SparkSession, types, functions

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+

spark = SparkSession.builder \
    .appName('EntityResolution') \
    .getOrCreate()
spark.conf.set("spark.executor.memory", "8g")
spark.conf.set("spark.driver.memory", "8g")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
spark.sparkContext.setLogLevel('WARN')
assert spark.version >= '2.3'  # make sure we have Spark 2.3+


@functions.udf(returnType=types.StringType())
def get_jaccard_score_udf(amazon_keyword_list, google_keyword_list):
    length_of_intersection = len(set(amazon_keyword_list) & set(google_keyword_list))
    length_of_union = len(set(amazon_keyword_list) | set(google_keyword_list))
    return float(length_of_intersection / length_of_union)


@functions.udf(returnType=types.StringType())
def get_street_name_udf(street_address):
    street_address = street_address.split('|')[0].split(',')[0]
    return street_address


class EntityResolution:
    def __init__(self, dataFile1, dataFile2):
        self.realtor_df_first = spark.read.csv(dataFile1, header=True)
        self.tax_df_first = spark.read.csv(dataFile2, header=True)

        self.realtor_df_first = self.realtor_df_first.withColumn(
            "STREET_NAME", get_street_name_udf(self.realtor_df_first["address"])
        )

    def preprocessDF(self, df, cols):
        df = df.withColumn('joined_columns', functions.lower(df[cols[0]]))

        regex_tokenizer = RegexTokenizer(inputCol="joined_columns", outputCol="joinKey", pattern=r'\W+')

        df = regex_tokenizer.transform(df)
        return df

    def filtering(self, df1, df2):
        realtor_df = df1.withColumn("keywords", functions.explode(df1['joinKey'])) \
            .selectExpr('*', "joinKey AS joinKey1", "keywords")
        tax_df = df2.withColumn("keywords", functions.explode(df2['joinKey'])) \
            .selectExpr('*', "joinKey AS joinKey2", "keywords")

        can_df = realtor_df.join(tax_df, realtor_df['keywords'] == tax_df['keywords']) \
            .drop('keywords') \
            .dropDuplicates()

        return can_df

    def verification(self, cand_df_arg, threshold):
        result_df = cand_df_arg \
            .withColumn('jaccard', get_jaccard_score_udf(cand_df_arg['joinKey1'], cand_df_arg['joinKey2']))
        result_df = result_df.filter(result_df['jaccard'] >= threshold)
        return result_df

    def evaluate(self, result_arg, ground_truth_arg):
        t_calculated = len(set(result_arg) & set(ground_truth_arg))
        r_calculated = len(result_arg)
        a_calculated = len(ground_truth_arg)

        try:
            precision = t_calculated / r_calculated
            recall = t_calculated / a_calculated
            f_measure = (2 * precision * recall) / (precision + recall)
        except ZeroDivisionError:
            precision = 0.0
            recall = 0.0
            f_measure = 0.0
            print('precision recall f_measure might be zero')

        return precision, recall, f_measure

    def jaccardJoin(self, cols1, cols2, threshold):
        realtor_df = self.preprocessDF(self.realtor_df_first, cols1)
        tax_df = self.preprocessDF(self.tax_df_first, cols2)
        print("Before filtering: %d pairs in total" % (self.realtor_df_first.count() * self.tax_df_first.count()))

        cand_df = self.filtering(realtor_df, tax_df)
        # print("After Filtering: %d pairs left" % (cand_df.count()))

        result_df = self.verification(cand_df, threshold)
        # print("After Verification: %d similar pairs" % (result_df.count()))

        return result_df


if __name__ == "__main__":
    er = EntityResolution("realtor.csv", "property_tax_report_2019.csv")
    realtor_cols = ["STREET_NAME"]
    tax_cols = ["ADDRESS"]
    resultDF = er.jaccardJoin(realtor_cols, tax_cols, 0.5)
    realtor_simplified_df = resultDF.dropDuplicates(['mls'])
    realtor_simplified_df.toPandas().to_csv("realtor_simplified.csv")

