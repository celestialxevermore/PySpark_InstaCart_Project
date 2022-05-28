# Databricks notebook source
# MAGIC %md
# MAGIC ### Machine Learning Advanced Big data Project

# COMMAND ----------

orders_sdf = spark.read.csv('/FileStore/tables/orders.csv', header=True, inferSchema=True)
trains_sdf = spark.read.csv('/FileStore/tables/order_products_train.csv', header=True, inferSchema=True)
products_sdf = spark.read.csv('/FileStore/tables/products.csv', header=True, inferSchema=True)
aisles_sdf = spark.read.csv('/FileStore/tables/aisles.csv', header=True, inferSchema=True)
depts_sdf = spark.read.csv('/FileStore/tables/departments.csv', header=True, inferSchema=True)

# COMMAND ----------

# MAGIC %fs 
# MAGIC cp /FileStore/tables/order_products_prior.zip file:/home/order_products_prior.zip 

# COMMAND ----------

import pandas as pd

priors_pdf = pd.read_csv('/home/order_products_prior.zip', compression='zip', header=0, sep=',', quotechar='"')
priors_sdf = spark.createDataFrame(priors_pdf)
del priors_pdf # 메모리 절약을 위해 pandas dataframe삭제

# COMMAND ----------

orders_sdf.createOrReplaceTempView("orders")
priors_sdf.createOrReplaceTempView("priors")
trains_sdf.createOrReplaceTempView("trains")
products_sdf.createOrReplaceTempView("products")
aisles_sdf.createOrReplaceTempView("aisles")
depts_sdf.createOrReplaceTempView("depts")

# COMMAND ----------

#테이블 등록
spark.catalog.listTables()

# COMMAND ----------



# COMMAND ----------

# MAGIC %fs
# MAGIC rm -r dbfs:/user/hive/warehouse/order_priors_prods

# COMMAND ----------

# MAGIC %sql 
# MAGIC drop table if exists order_priors_prods; 
# MAGIC 
# MAGIC -- priors와 orders를 조인 
# MAGIC -- orders에서는 pk를 확인할 수 없기 때문에 조인
# MAGIC create table  order_priors_prods 
# MAGIC as 
# MAGIC select a.order_id, a.product_id, a.add_to_cart_order, a.reordered, b.user_id, b.eval_set, b.order_number, b.order_dow, b.order_hour_of_day, b.days_since_prior_order 
# MAGIC from priors a, orders b 
# MAGIC where a.order_id = b.order_id;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating a product analysis table based on product level analysis attributes
# MAGIC * PK is a product code (product_id) and generates a product analysis table with attributes analyzed in the previous EDA..

# COMMAND ----------

# MAGIC %fs
# MAGIC rm -r dbfs:/user/hive/warehouse/prd_mart

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists prd_mart;
# MAGIC 
# MAGIC create table prd_mart
# MAGIC as
# MAGIC with 
# MAGIC -- with 구문 첫번째 집합. product_id 레벨로 group by 하여 상품별 서로 다른 개별 사용자 비율을 추출한 결과에 상품명과 상품 중분류명 알기 위해 products와 aisles로 조인
# MAGIC order_prods_grp as
# MAGIC (
# MAGIC   select a.product_id 
# MAGIC     -- ## 상품별 재주문 속성
# MAGIC     , sum(case when reordered=1 then 1 else 0 end) as prd_reordered_cnt -- 상품별 재 주문 건수
# MAGIC     , sum(case when reordered=0 then 1 else 0 end) as prd_no_reordered_cnt -- 상품별 재 주문 하지 않은 건수 
# MAGIC     , avg(reordered) prd_avg_reordered -- 상품별 재 주문 비율
# MAGIC     -- ## 상품별 고유 사용자 및 이전 주문이후 걸린 일자 속성. 
# MAGIC     , count(distinct user_id) prd_unq_usr_cnt -- 상품별 고유 사용자 건수
# MAGIC     , count(*)  prd_total_cnt -- 상품별 건수
# MAGIC     , count(distinct user_id)/count(*) as prd_usr_ratio -- 상품별 전체 건수 대비 고유 사용자 비율
# MAGIC     , max(c.aisle_id) aisle_id -- 상품 중분류 코드 
# MAGIC     , nvl(avg(days_since_prior_order), 0) as prd_avg_prior_days -- 평균 이전 주문이후 걸린 일자, null인 경우 0으로 변환. 
# MAGIC     , nvl(min(days_since_prior_order), 0) as prd_min_prior_days -- 최소 이전 주문이후 걸린 일자, null인 경우 0으로 변환. 
# MAGIC     , nvl(max(days_since_prior_order), 0) as prd_max_prior_days -- 최대 이전 주문이후 걸린 일자, null인 경우 0으로 변환. 
# MAGIC     from order_priors_prods a, products b, aisles c
# MAGIC   where a.product_id = b.product_id 
# MAGIC   and b.aisle_id = c.aisle_id
# MAGIC   group by a.product_id
# MAGIC ),
# MAGIC -- with 구문 두번째 집합. product_id 레벨로 group by 하여 상품별 서로 다른 개별 사용자 비율을 추출한 결과에 product_name과 중분류명, 대분류명을 알기 위해 aisles와 dept로 조인. 
# MAGIC order_aisles_grp as
# MAGIC (
# MAGIC   select c.aisle_id as aisle_id 
# MAGIC      , count(distinct a.user_id) aisle_distinct_usr_cnt -- 상품 중분류별 고유 사용자 건수
# MAGIC      , count(*)  aisle_total_cnt -- 상품 중분류 건수
# MAGIC      , count(distinct a.user_id)/count(*) as aisle_usr_ratio -- 상품 중분류 건수 대비 고유 사용자 건수 비율
# MAGIC   from order_priors_prods a, products b, aisles c
# MAGIC   where a.product_id = b.product_id 
# MAGIC   and b.aisle_id = c.aisle_id
# MAGIC   group by c.aisle_id
# MAGIC ),
# MAGIC -- with 구문 세번째 집합. 상품 중분류 별 개별 사용자 비율과 상품별 개별 사용자 비율 차이 추출. 
# MAGIC order_prd_grp_aisle as
# MAGIC (
# MAGIC   select product_id, prd_reordered_cnt,  prd_no_reordered_cnt, prd_avg_reordered, prd_unq_usr_cnt, prd_total_cnt, prd_usr_ratio
# MAGIC     , prd_avg_prior_days, prd_min_prior_days, prd_max_prior_days-- 상품별 속성들
# MAGIC     , b.aisle_distinct_usr_cnt, b.aisle_total_cnt, b.aisle_usr_ratio -- 상품 중분류별 속성들 
# MAGIC     , a.prd_usr_ratio - b.aisle_usr_ratio as usr_ratio_diff -- 상품 중분류 별 개별 사용자 비율과 상품별 개별 사용자 비율 차이
# MAGIC   from order_prods_grp a, order_aisles_grp b
# MAGIC   where a.aisle_id = b.aisle_id
# MAGIC ) 
# MAGIC -- end of with 절
# MAGIC select * from order_prd_grp_aisle

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from prd_mart limit 10

# COMMAND ----------

# MAGIC %sql
# MAGIC --49676
# MAGIC select count(*) from prd_mart

# COMMAND ----------

import pyspark.sql.functions as F

prd_mart_sdf = spark.sql("select * from prd_mart")

display(prd_mart_sdf.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in prd_mart_sdf.columns]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create user analysis tables based on user-level analysis properties
# MAGIC * PK is the user ID (user_id) and creates a user analysis table with attributes analyzed by previous EDA.
# MAGIC * Order_id is required to create prediction data in the future. To this end, it is necessary to extract order_id by joining the order data for training and testing with user_id.
# MAGIC * The order table is user_id level m, but if eval_set is train and test, the user_id level is 1, so the user_mart table level does not change when joining.

# COMMAND ----------

# MAGIC %fs
# MAGIC rm -r dbfs:/user/hive/warehouse/user_mart_01

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists user_mart_01;
# MAGIC 
# MAGIC create table user_mart_01
# MAGIC as
# MAGIC select user_id 
# MAGIC   , count(*) as usr_total_cnt -- 사용자별 주문 건수
# MAGIC   -- 주문 건수 관련 속성 추출. 
# MAGIC   , count(distinct product_id) prd_uq_cnt  -- 사용자별 고유 상품 주문 건수
# MAGIC   , count(distinct order_id) order_uq_cnt -- 사용자별 고유 주문 건수
# MAGIC   , count(*)/count(distinct order_id) as usr_avg_prd_cnt -- 사용자별 1회 주문시 평균 주문 상품 건수
# MAGIC   , count(*)/count(distinct product_id) as usr_avg_uq_prd_cnt -- 사용자별 1회 주문시 평균 고유 주문 상품 건수
# MAGIC   , count(distinct product_id)/count(*) as usr_uq_prd_ratio --사용자별 총 상품 건수 대비 고유 상품 건수 비율
# MAGIC   -- ### reordered 관련 속성 추출. ###
# MAGIC   , sum(reordered) usr_reord_cnt -- 사용자별 reordered된 상품 건수
# MAGIC   , sum(case when reordered = 0 then 1 else 0 end) as usr_no_reord_cnt -- 사용자별 reorder 하지 않은 상품 건수. count(*) - sum(reoredred)와 동일. 
# MAGIC   , avg(reordered) usr_reordered_avg -- 사용자별 reordered 비율
# MAGIC   -- ### days_since_prior_order 관련 속성 추출. ###
# MAGIC   , avg(days_since_prior_order) usr_avg_prior_days
# MAGIC   , max(days_since_prior_order) usr_max_prior_days
# MAGIC   , min(days_since_prior_order) usr_min_prior_days
# MAGIC   -- ### order_dow, order_hour_of_day 관련 속성 추출. ###
# MAGIC   , avg(order_dow) usr_avg_order_dow
# MAGIC   , avg(order_hour_of_day) usr_avg_order_hour_of_day
# MAGIC   -- 사용자별 최대 order_number
# MAGIC   , max(order_number) as usr_max_order_number
# MAGIC from order_priors_prods a group by user_id

# COMMAND ----------

# MAGIC %sql 
# MAGIC select count(*) from user_mart_01

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists user_mart;
# MAGIC -- cmd 20 
# MAGIC -- orders는 eval_set이 train/test일 경우 한개의 user_id가 한개의 order_id를 가짐. 때문에 train/test인 경우 조인키값 user_id로 1레벨이 됨.
# MAGIC -- order_priors_prods에 있는 모든 user_id는 orders의 모든 user_id와 동일. orders는 user_id별로 여러건의 order가 있고, 이들중 마지막 order를 train또는 test로 할당하기 때문
# MAGIC -- 따라서 user_mart_01과 eval_set이 train과 test인 orders를 user_id로 조인하면 1:1 조인이 되고 user_mart_01의 집합 레벨의 변화가 없음. outer join을 하지 않아도 됨. 
# MAGIC create table user_mart
# MAGIC as
# MAGIC select a.*, b.order_id, b.eval_set, b.days_since_prior_order
# MAGIC from user_mart_01 a, orders b
# MAGIC where a.user_id = b.user_id
# MAGIC and b.eval_set in ('train', 'test')   

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from user_mart limit 10

# COMMAND ----------

# MAGIC %sql
# MAGIC --206209
# MAGIC select count(*) from user_mart

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*)
# MAGIC from orders b
# MAGIC where b.eval_set in ('train', 'test')  

# COMMAND ----------

# MAGIC %md 
# MAGIC user_id : 1 means that the tables includes records for training 
# MAGIC 
# MAGIC user_id : 3 means that the tables includes records for test

# COMMAND ----------

# MAGIC 
# MAGIC %sql
# MAGIC select * from orders where user_id = 1

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from orders where user_id = 3

# COMMAND ----------

# MAGIC %md
# MAGIC ### 사용자 + 상품 레벨의 분석 속성에 기반한 사용자+상품 분석 테이블 생성
# MAGIC * PK 는 사용자아이디(user_id)+상품코드(product_id)이며 이전 EDA에서 분석한 속성들로 사용자+상품 분석 테이블 생성 테이블 생성.
# MAGIC * 앞에서 만든 prd_mart, user_mart를 사용자+상품 분석 테이블과 조인하여 상품관련 속성, 사용자 관련 속성을 결합함.

# COMMAND ----------

# MAGIC %fs
# MAGIC rm -r dbfs:/user/hive/warehouse/up_mart

# COMMAND ----------

# MAGIC %fs
# MAGIC rm -r dbfs:/user/hive/warehouse/up_mart_01

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from order_priors_prods where user_id = 1

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists up_mart;
# MAGIC drop table if exists up_mart_01;
# MAGIC 
# MAGIC create table up_mart
# MAGIC as
# MAGIC with 
# MAGIC -- 사용자+상품 레벨로 group by 하여 속성 추출. 
# MAGIC up_grp as
# MAGIC (
# MAGIC SELECT user_id, product_id
# MAGIC     , count(*) up_cnt  -- 사용자의 개별 상품별 주문 건수
# MAGIC     , sum(reordered) up_reord_cnt -- 사용자의 개별 상품별 reorder 건수
# MAGIC     , sum(case when reordered=0 then 1 else 0 end) up_no_reord_cnt
# MAGIC     , avg(reordered) up_reoredered_avg -- 사용자의 개별 상품 주문별 reorder비율 
# MAGIC     , max(order_number) up_max_ord_num -- 사용자+상품레벨에서 가장 큰 order_number. order_number는 사용자 별로 주문을 수행한 일련번호를 순차적으로 가짐. 
# MAGIC     , min(order_number) up_min_ord_num -- 사용자+상품레벨에서 가장 작은 order_number
# MAGIC     , avg(add_to_cart_order) up_avg_cart --사용자 상품레벨에서 보통 cart에 몇번째로 담는가?
# MAGIC     , avg(days_since_prior_order) as up_avg_prior_days
# MAGIC     , max(days_since_prior_order) as up_max_prior_days
# MAGIC     , min(days_since_prior_order) as up_min_prior_days
# MAGIC     , avg(order_dow) as up_avg_ord_dow
# MAGIC     , avg(order_hour_of_day) as up_avg_ord_hour
# MAGIC FROM order_priors_prods GROUP BY user_id, product_id
# MAGIC )
# MAGIC -- end of with 절 
# MAGIC -- 사용자 레벨로 group by 한 user_mart 테이블과 조인하여 사용자 레벨 속성과 사용자+상품 레벨 속성의 비율을 추출. 
# MAGIC select a.* 
# MAGIC   , a.up_cnt/b.usr_total_cnt as up_usr_ratio -- 사용자별 전체 주문 건수 대비 사용자+상품 주문 건수 비율
# MAGIC   , a.up_reord_cnt/b.usr_reord_cnt as up_usr_reord_ratio -- 사용자별 전체 재주문 건수 대비 사용자+상품 재주문 건수 비율
# MAGIC   , b.usr_reord_cnt
# MAGIC   , b.usr_max_order_number - a.up_max_ord_num as up_usr_ord_num_diff -- 사용자의 가장 최근 주문(가장 큰 주문번호)에서 현 상품 주문번호가 어느정도 이후에 있는지
# MAGIC from up_grp a, user_mart b
# MAGIC where a.user_id = b.user_id

# COMMAND ----------

# MAGIC %sql
# MAGIC --13307953
# MAGIC select count(*) from up_mart

# COMMAND ----------

# MAGIC %sql
# MAGIC -- up_mart에서 user_mart로, user_id로 join이 안되거나 prd_mart로, product_id로 join이 안되는 경우 추출.  
# MAGIC select count(*)
# MAGIC from up_mart a 
# MAGIC left outer join user_mart b
# MAGIC on a.user_id = b.user_id
# MAGIC left outer join prd_mart c
# MAGIC on a.product_id = c.product_id
# MAGIC where (b.user_id is null or c.product_id is null)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from aisles where aisle_id='Blunted'
# MAGIC /* 
# MAGIC select * from products a where product_id = 6816
# MAGIC select * from aisles where aisle_id='Blunted' 
# MAGIC */

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 현재까지 만들어진 테이블의 건수 조사 
# MAGIC select 'user_mart count' as gubun, count(*) from user_mart
# MAGIC union all
# MAGIC select 'prd_mart count' as gubun, count(*) from prd_mart
# MAGIC union all
# MAGIC select 'up_mart count' as gubun, count(*) from up_mart

# COMMAND ----------

# MAGIC %md
# MAGIC #### 현재까지 만든 prd_mart, user_mart, up_mart를 결합하여 data_mart 생성. 
# MAGIC * 생성된 data_mart는 up_mart를 기준으로 prd_mart, user_mart를 조인하여 상품 분석속성, 사용자 분석속성을 결합.

# COMMAND ----------

# MAGIC %sql
# MAGIC describe up_mart

# COMMAND ----------

print(spark.sql("select * from up_mart").columns)
print(spark.sql("select * from user_mart").columns)
print(spark.sql("select * from prd_mart").columns)

# COMMAND ----------

# MAGIC %fs
# MAGIC rm -r dbfs:/user/hive/warehouse/data_mart

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists data_mart;
# MAGIC 
# MAGIC -- up_mart에 user_mart를 user_id로 조인, prd_mart는 product_id로 조인하여 개별 xxx_mart테이블의 속성들을 취합하여 data_mart 테이블 생성. 약 4분정도 걸림. 
# MAGIC create table data_mart
# MAGIC as
# MAGIC select 
# MAGIC   -- up_mart 컬럼 
# MAGIC   a.user_id, a.product_id, b.order_id -- 테스트 데이터 예측 데이터 제출을 위해서 order_id가 필요함. 
# MAGIC   , up_cnt, up_reord_cnt, up_no_reord_cnt, up_reoredered_avg, up_max_ord_num, up_min_ord_num, up_avg_cart, up_avg_prior_days, up_max_prior_days
# MAGIC   , up_min_prior_days, up_avg_ord_dow, up_avg_ord_hour, up_usr_ratio, up_usr_reord_ratio, up_usr_ord_num_diff
# MAGIC   -- user_mart 컬럼, eval_set에 train과 test용 데이터(사용자)구분
# MAGIC   , usr_total_cnt, prd_uq_cnt, order_uq_cnt, usr_avg_prd_cnt, usr_avg_uq_prd_cnt, usr_uq_prd_ratio, a.usr_reord_cnt, usr_no_reord_cnt, usr_reordered_avg, usr_avg_prior_days
# MAGIC   , usr_max_prior_days, usr_min_prior_days, usr_avg_order_dow, usr_avg_order_hour_of_day, usr_max_order_number, eval_set, days_since_prior_order
# MAGIC   -- prd_mart 컬럼
# MAGIC   , prd_reordered_cnt, prd_no_reordered_cnt, prd_avg_reordered, prd_unq_usr_cnt, prd_total_cnt, prd_usr_ratio, prd_avg_prior_days, prd_min_prior_days, prd_max_prior_days
# MAGIC   , aisle_distinct_usr_cnt, aisle_total_cnt, aisle_usr_ratio, usr_ratio_diff
# MAGIC from up_mart a, user_mart b, prd_mart c
# MAGIC where a.user_id = b.user_id and a.product_id = c.product_id

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 현재까지 생성된 테이블의 건수 조사. data_mart는 up_mart와 동일 건수 - 3 
# MAGIC select 'data_mart count' as gubun, count(*) from data_mart
# MAGIC union all 
# MAGIC select 'user_mart count' as gubun, count(*) from user_mart
# MAGIC union all
# MAGIC select 'prd_mart count' as gubun, count(*) from prd_mart
# MAGIC union all
# MAGIC select 'up_mart count' as gubun, count(*) from up_mart

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from data_mart limit 10

# COMMAND ----------

# MAGIC %md
# MAGIC ### 학습과 테스트용 데이터 세트 생성. 
# MAGIC * order_products_train.csv(trains 테이블)는  train용으로 reordered label 값이 주어져 있음.
# MAGIC * trains 테이블의 pk는 order_id + product_id 이지만 실제로는 1개의 user_id에 1개의 order_id만 할당되므로 user_id + product_id로 unique함. 
# MAGIC * trains 테이블과 orders 테이블을 조인하여 user_id를 가져오는 order_trains_prods 테이블 생성. 
# MAGIC * order_trains_prods 테이블을 기준으로 data_mart에서 생성한 속성을 붙이려고 두개의 테이블을 user_id + product_id로 조인(order_trains_prods 레프트 아우터 조인)하면 많은 데이터가 조인되지 않음.  
# MAGIC * 조인되지 않을 경우에 data_mart에서 생성한 속성을 사용할 수 없음. 
# MAGIC * data_mart를 기준으로 order_trains_prods를 조인(data_mart 레프트 아우터 조인)하여 label값인 reordered를 설정하고 조인되지 않는 경우 reordered를 0으로 설정.

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from trains limit 10

# COMMAND ----------

# MAGIC %sql
# MAGIC --1384617
# MAGIC select count(*) from trains

# COMMAND ----------

# MAGIC %fs
# MAGIC rm -r dbfs:/user/hive/warehouse/order_trains_prods

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists order_trains_prods;
# MAGIC -- order_products_train 데이터에(trains 테이블)에 user_id를 얻기 위해서 orders 테이블과 조인
# MAGIC -- 해당 테이블은 kaggle에서 train 용으로 제공었지만, 많은 속성(feature)로 만들어진 data_mart 테이블에 비해 적은 속성을 가지고 있음. 
# MAGIC create table order_trains_prods
# MAGIC as
# MAGIC select a.order_id, a.product_id, a.reordered
# MAGIC   , b.user_id
# MAGIC from trains a, orders b
# MAGIC where a.order_id = b.order_id

# COMMAND ----------


