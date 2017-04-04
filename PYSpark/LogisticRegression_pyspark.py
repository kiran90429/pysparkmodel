from pyspark import SparkConf,SparkContext

import datetime;

#from pyspark.sql.functions import stddev_pop,avg


from pyspark.sql import SQLContext
from pyspark.ml.feature import StandardScaler
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS,\
    LogisticRegressionModel, LogisticRegressionWithSGD

def parsePoint(line):
    
    #values = [float(x) for x in line.split(',')]
    values = line.values
    return LabeledPoint(values[0],values[1:]);


def giveMeMonth(date):
    year,month,date = (int(x) for x in date.split('-'))
    ans = datetime.date(year,month,date);
    month = ans.strftime("%m");
    return int(month);

def SetConditions(column):
    
    if(column['final_delay'] > 14):
        return 1;
    
    else:
        return 0;
    
        




fpcsvfile = "C:\Users\kiran.kandula\workspace\largefiles\\fp_acctdocheader_fleetpride_open_last500kto100k.csv";

sparkConf = SparkConf().setAppName("FPDATA");

sc = SparkContext(conf = sparkConf);

sqlContext = SQLContext(sc);



df_fpdata = sqlContext.read.load(fpcsvfile,format='com.databricks.spark.csv',header='true',delimiter=',');

df_fpdata.printSchema();

df_fpdata = df_fpdata\
.withColumn('company_code',df_fpdata.company_code.cast('int'))\
.withColumn('doctype',df_fpdata.doctype.cast('int'))\
.withColumn('branch',df_fpdata.branch.cast('int'))\
.withColumn('invoice_amount',df_fpdata.invoice_amount.cast('int'))\
.withColumn('isOpen',df_fpdata.isOpen.cast('int'))\
.withColumn('reference',df_fpdata.reference.cast('int'))\
.withColumn('ship_to',df_fpdata.ship_to.cast('int'))\
.withColumn('payment_terms',df_fpdata.payment_terms.cast('int'))\
.withColumn('error_code_id',df_fpdata.error_code_id.cast('int'))\
.withColumn('due_date',df_fpdata.due_date.cast('date'))\
.withColumn('update_date',df_fpdata.update_date.cast('date'));
#df_fpdata.printSchema();

#df_fpdata.show(10);

sqlContext.registerDataFrameAsTable(df_fpdata, 'fp');

query = 'select * from fp limit 25000';

print df_fpdata.printSchema();


df_query_result = sqlContext.sql(query);

#converting to pandas;


df_fpdata_pandas = df_query_result.toPandas();

df_fpdata_pandas =  df_fpdata_pandas[['branch','company_code','doctype','invoice_amount','due_date','update_date','isOpen','reference','ship_to','payment_terms','error_code_id']];

#print df_fpdata_pandas;

df_fpdata_pandas.fillna(0,inplace=True);

#print df_fpdata_pandas;
df_fpdata_pandas["month"] = df_fpdata_pandas["due_date"].apply(lambda x:giveMeMonth(str(x)));

df_fpdata_pandas["month"] = df_fpdata_pandas["month"].apply(lambda x:int(x));

#choosing only the ones for which the payment is done;

df_fpdata_pandas = df_fpdata_pandas[df_fpdata_pandas["isOpen"] == 0];

#df_fpdata_pandas["due_date"] = df_fpdata_pandas["due_date"].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d").strftime("%Y%m%d"));

#df_fpdata_pandas["update_date"] = df_fpdata_pandas["update_date"].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d").strftime("%Y%m%d"));

df_fpdata_pandas["doctype"] = df_fpdata_pandas["doctype"].apply(lambda x:int(x));

#df_fpdata_pandas[""]

#print df_fpdata_pandas;


df_fpdata_pandas['final_delay'] = 0;

df_fpdata_pandas['final_delay'] = [((datetime.datetime.strptime(str(start),"%Y-%m-%d")) - (datetime.datetime.strptime(str(end),"%Y-%m-%d"))).days for start,end in zip(df_fpdata_pandas['update_date'],df_fpdata_pandas['due_date'])];

print df_fpdata_pandas;
print df_fpdata_pandas['final_delay'];


df_fpdata_pandas["final_labels"] = 0;

df_fpdata_pandas = df_fpdata_pandas.assign(final_labels = df_fpdata_pandas.apply(SetConditions,axis=1));

print df_fpdata_pandas;

print df_fpdata_pandas['final_labels'];


label_0 = df_fpdata_pandas[df_fpdata_pandas["final_labels"]==0];
    
label_1 = df_fpdata_pandas[df_fpdata_pandas["final_labels"]==1];
    
    #i = 0;
    
    
df_final_0 = df_fpdata_pandas[df_fpdata_pandas['final_labels']== 0]
     
df_final_0 = df_fpdata_pandas[-len(label_1):];
    
    #print df_final_0;   
    
    #print len(df_final_0);
    
df_final_1 = label_1;

df_final = df_final_0.append(df_final_1);

print df_final;
   

df_trainingData = df_final[["final_labels","company_code","branch","doctype","invoice_amount","ship_to","reference","payment_terms","error_code_id","month"]]


data = df_trainingData.apply(parsePoint,axis=1);

print data;

# spark_df = sqlContext.createDataFrame(data);
# 
# spark_data = spark_df.rdd;


#df_trainingData = df_trainingData.take(100);
 
#spark_df_train = sqlContext.createDataFrame(df_trainingData);
# 
# spark_df = spark_df_train.limit(100);
# 
# parseddata = spark_df_train.rdd.map(parsePoint);
# 
# print parseddata;

#parseddata = parseddata.LIMIT(100);

#print parseddata;

#parseddata = parseddata.rdd.limit(100);

model = LogisticRegressionWithSGD.train(sc.parallelize(data).cache(),iterations=100);

# Evaluating the model on training data
labelsAndPreds = data.map(lambda p: (p.label, model.predict(p.features)))
print labelsAndPreds;
trainErr = filter(lambda (v, p): v != p,labelsAndPreds);
print trainErr;
# 
# error = float(len(trainErr))/float(len(data));
# 
# print "error----",error;
# print("Training Error = " + str(trainErr))





# print df_features;
# 
# df_labels = df_final["final_labels"];
# 
# print df_labels;
# 
# 
# 
# df_final.select(["final_labels","company_code","branch"]).rdd.map(lambda row:LabeledPoint(row.final_labels,row.company_code,row.branch));
# 
# 
# # #normalization:
# # 
# # standardizer = StandardScaler();
# # 
# # model =  standardizer.fit(df_features);
# # 
# # features_transform = model.transform(df_features);
# # 
# # features_transform.take(2);











