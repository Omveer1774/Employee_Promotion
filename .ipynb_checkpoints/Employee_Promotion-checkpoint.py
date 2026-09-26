{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "dde3992f-46f0-48b6-9264-efc8854d43c7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import keras\n",
    "import tensorflow as tf\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "import keras.models as km\n",
    "from keras.models import Sequential\n",
    "from keras import activations, initializers, regularizers, constraints\n",
    "from keras.layers import Dense, Activation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "78e8a199-a6d2-4834-82fe-52e8840cd8dc",
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset=pd.read_csv(\"Employee_Promotion_Dataset.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "ed8292d1-299e-4674-8604-2afb6890b45b",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>EmployeeID</th>\n",
       "      <th>Department</th>\n",
       "      <th>Education</th>\n",
       "      <th>Gender</th>\n",
       "      <th>RecruitmentChannel</th>\n",
       "      <th>NoOfTrainings</th>\n",
       "      <th>Age</th>\n",
       "      <th>PreviousYearRating</th>\n",
       "      <th>LengthOfService</th>\n",
       "      <th>KPIsMetAbove80Percent</th>\n",
       "      <th>AwardsWon</th>\n",
       "      <th>AvgTrainingScore</th>\n",
       "      <th>MonthlySalary</th>\n",
       "      <th>Promoted</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>10001</td>\n",
       "      <td>Technology</td>\n",
       "      <td>Bachelor</td>\n",
       "      <td>Male</td>\n",
       "      <td>Other</td>\n",
       "      <td>1</td>\n",
       "      <td>51</td>\n",
       "      <td>2</td>\n",
       "      <td>9</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>68</td>\n",
       "      <td>87022</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>10002</td>\n",
       "      <td>Operations</td>\n",
       "      <td>Bachelor</td>\n",
       "      <td>Female</td>\n",
       "      <td>Sourcing</td>\n",
       "      <td>2</td>\n",
       "      <td>54</td>\n",
       "      <td>5</td>\n",
       "      <td>14</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>41</td>\n",
       "      <td>78802</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>10003</td>\n",
       "      <td>Marketing</td>\n",
       "      <td>PhD</td>\n",
       "      <td>Female</td>\n",
       "      <td>Sourcing</td>\n",
       "      <td>2</td>\n",
       "      <td>57</td>\n",
       "      <td>3</td>\n",
       "      <td>12</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>45</td>\n",
       "      <td>103165</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>10004</td>\n",
       "      <td>Technology</td>\n",
       "      <td>Bachelor</td>\n",
       "      <td>Female</td>\n",
       "      <td>Referred</td>\n",
       "      <td>2</td>\n",
       "      <td>51</td>\n",
       "      <td>5</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>73</td>\n",
       "      <td>118069</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>10005</td>\n",
       "      <td>Sales</td>\n",
       "      <td>PhD</td>\n",
       "      <td>Male</td>\n",
       "      <td>Sourcing</td>\n",
       "      <td>1</td>\n",
       "      <td>47</td>\n",
       "      <td>2</td>\n",
       "      <td>8</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>51</td>\n",
       "      <td>98594</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   EmployeeID  Department Education  Gender RecruitmentChannel  NoOfTrainings  \\\n",
       "0       10001  Technology  Bachelor    Male              Other              1   \n",
       "1       10002  Operations  Bachelor  Female           Sourcing              2   \n",
       "2       10003   Marketing       PhD  Female           Sourcing              2   \n",
       "3       10004  Technology  Bachelor  Female           Referred              2   \n",
       "4       10005       Sales       PhD    Male           Sourcing              1   \n",
       "\n",
       "   Age  PreviousYearRating  LengthOfService  KPIsMetAbove80Percent  AwardsWon  \\\n",
       "0   51                   2                9                      1          0   \n",
       "1   54                   5               14                      0          0   \n",
       "2   57                   3               12                      1          0   \n",
       "3   51                   5                4                      0          0   \n",
       "4   47                   2                8                      0          0   \n",
       "\n",
       "   AvgTrainingScore  MonthlySalary  Promoted  \n",
       "0                68          87022         1  \n",
       "1                41          78802         0  \n",
       "2                45         103165         1  \n",
       "3                73         118069         0  \n",
       "4                51          98594         0  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "6bd948a5-6f32-4a07-9c83-28f25efa659d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>EmployeeID</th>\n",
       "      <th>Department</th>\n",
       "      <th>Education</th>\n",
       "      <th>Gender</th>\n",
       "      <th>RecruitmentChannel</th>\n",
       "      <th>NoOfTrainings</th>\n",
       "      <th>Age</th>\n",
       "      <th>PreviousYearRating</th>\n",
       "      <th>LengthOfService</th>\n",
       "      <th>KPIsMetAbove80Percent</th>\n",
       "      <th>AwardsWon</th>\n",
       "      <th>AvgTrainingScore</th>\n",
       "      <th>MonthlySalary</th>\n",
       "      <th>Promoted</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>4995</th>\n",
       "      <td>14996</td>\n",
       "      <td>Finance</td>\n",
       "      <td>PhD</td>\n",
       "      <td>Female</td>\n",
       "      <td>Referred</td>\n",
       "      <td>3</td>\n",
       "      <td>42</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>46</td>\n",
       "      <td>158257</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4996</th>\n",
       "      <td>14997</td>\n",
       "      <td>Sales</td>\n",
       "      <td>PhD</td>\n",
       "      <td>Female</td>\n",
       "      <td>Sourcing</td>\n",
       "      <td>4</td>\n",
       "      <td>42</td>\n",
       "      <td>5</td>\n",
       "      <td>9</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>66</td>\n",
       "      <td>106571</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4997</th>\n",
       "      <td>14998</td>\n",
       "      <td>Sales</td>\n",
       "      <td>High School</td>\n",
       "      <td>Male</td>\n",
       "      <td>Referred</td>\n",
       "      <td>2</td>\n",
       "      <td>24</td>\n",
       "      <td>1</td>\n",
       "      <td>16</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>87</td>\n",
       "      <td>116497</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4998</th>\n",
       "      <td>14999</td>\n",
       "      <td>Finance</td>\n",
       "      <td>Bachelor</td>\n",
       "      <td>Female</td>\n",
       "      <td>Sourcing</td>\n",
       "      <td>3</td>\n",
       "      <td>49</td>\n",
       "      <td>2</td>\n",
       "      <td>12</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>86</td>\n",
       "      <td>145013</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4999</th>\n",
       "      <td>15000</td>\n",
       "      <td>Technology</td>\n",
       "      <td>Bachelor</td>\n",
       "      <td>Male</td>\n",
       "      <td>Other</td>\n",
       "      <td>2</td>\n",
       "      <td>46</td>\n",
       "      <td>4</td>\n",
       "      <td>7</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>94</td>\n",
       "      <td>119150</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      EmployeeID  Department    Education  Gender RecruitmentChannel  \\\n",
       "4995       14996     Finance          PhD  Female           Referred   \n",
       "4996       14997       Sales          PhD  Female           Sourcing   \n",
       "4997       14998       Sales  High School    Male           Referred   \n",
       "4998       14999     Finance     Bachelor  Female           Sourcing   \n",
       "4999       15000  Technology     Bachelor    Male              Other   \n",
       "\n",
       "      NoOfTrainings  Age  PreviousYearRating  LengthOfService  \\\n",
       "4995              3   42                   3                4   \n",
       "4996              4   42                   5                9   \n",
       "4997              2   24                   1               16   \n",
       "4998              3   49                   2               12   \n",
       "4999              2   46                   4                7   \n",
       "\n",
       "      KPIsMetAbove80Percent  AwardsWon  AvgTrainingScore  MonthlySalary  \\\n",
       "4995                      0          0                46         158257   \n",
       "4996                      0          0                66         106571   \n",
       "4997                      1          0                87         116497   \n",
       "4998                      0          0                86         145013   \n",
       "4999                      0          0                94         119150   \n",
       "\n",
       "      Promoted  \n",
       "4995         0  \n",
       "4996         0  \n",
       "4997         1  \n",
       "4998         0  \n",
       "4999         1  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "4c6395e0-d966-4421-ae0e-e8056b40562a",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "dataset.drop(['EmployeeID', 'Education','Department'], axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "b9939b51-7633-4edf-8392-2f235471eb29",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>NoOfTrainings</th>\n",
       "      <th>Age</th>\n",
       "      <th>PreviousYearRating</th>\n",
       "      <th>LengthOfService</th>\n",
       "      <th>KPIsMetAbove80Percent</th>\n",
       "      <th>AwardsWon</th>\n",
       "      <th>AvgTrainingScore</th>\n",
       "      <th>MonthlySalary</th>\n",
       "      <th>Promoted</th>\n",
       "      <th>RecruitmentChannel_Male</th>\n",
       "      <th>Gender_Referred</th>\n",
       "      <th>Gender_Sourcing</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>51</td>\n",
       "      <td>2</td>\n",
       "      <td>9</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>68</td>\n",
       "      <td>87022</td>\n",
       "      <td>1</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>54</td>\n",
       "      <td>5</td>\n",
       "      <td>14</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>41</td>\n",
       "      <td>78802</td>\n",
       "      <td>0</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>57</td>\n",
       "      <td>3</td>\n",
       "      <td>12</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>45</td>\n",
       "      <td>103165</td>\n",
       "      <td>1</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2</td>\n",
       "      <td>51</td>\n",
       "      <td>5</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>73</td>\n",
       "      <td>118069</td>\n",
       "      <td>0</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>47</td>\n",
       "      <td>2</td>\n",
       "      <td>8</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>51</td>\n",
       "      <td>98594</td>\n",
       "      <td>0</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   NoOfTrainings  Age  PreviousYearRating  LengthOfService  \\\n",
       "0              1   51                   2                9   \n",
       "1              2   54                   5               14   \n",
       "2              2   57                   3               12   \n",
       "3              2   51                   5                4   \n",
       "4              1   47                   2                8   \n",
       "\n",
       "   KPIsMetAbove80Percent  AwardsWon  AvgTrainingScore  MonthlySalary  \\\n",
       "0                      1          0                68          87022   \n",
       "1                      0          0                41          78802   \n",
       "2                      1          0                45         103165   \n",
       "3                      0          0                73         118069   \n",
       "4                      0          0                51          98594   \n",
       "\n",
       "   Promoted  RecruitmentChannel_Male  Gender_Referred  Gender_Sourcing  \n",
       "0         1                     True            False            False  \n",
       "1         0                    False            False             True  \n",
       "2         1                    False            False             True  \n",
       "3         0                    False             True            False  \n",
       "4         0                     True            False             True  "
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset_new = pd.get_dummies(dataset, ['RecruitmentChannel', 'Gender'], drop_first=True)\n",
    "dataset_new.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "3fd319c0-e5a7-43af-99c5-e10bd08e06de",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = dataset_new.drop(columns=[\"Promoted\"])\n",
    "y = dataset_new['Promoted']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "984e0cb4-9e21-4f46-87a7-41f9e003fb4c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Promoted\n",
       "0    2586\n",
       "1    2414\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset[\"Promoted\"].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "b2f0ccbe-0df5-4ae4-8ea0-713b873ed891",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state =123)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "287fa973-5fe3-4368-bb28-6629ae01f9d4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "11"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_X.shape[1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "e4b8628f-11e7-4e31-a382-0c4254314cd1",
   "metadata": {},
   "outputs": [],
   "source": [
    "sc = StandardScaler()\n",
    "train_X = sc.fit_transform(train_X)\n",
    "test_X = sc.transform(test_X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "c246058e-e9cb-4968-a81b-3a22a81f0a1b",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\lenovo\\anaconda3\\Lib\\site-packages\\keras\\src\\layers\\core\\dense.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
      "  super().__init__(activity_regularizer=activity_regularizer, **kwargs)\n"
     ]
    }
   ],
   "source": [
    "model = Sequential()\n",
    "model.add(Dense(units = 11, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))\n",
    "model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))\n",
    "model.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))\n",
    "model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "6eaf2530-77d7-4f96-84b0-e60f75f58534",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\">Model: \"sequential\"</span>\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1mModel: \"sequential\"\u001b[0m\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓\n",
       "┃<span style=\"font-weight: bold\"> Layer (type)                         </span>┃<span style=\"font-weight: bold\"> Output Shape                </span>┃<span style=\"font-weight: bold\">         Param # </span>┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩\n",
       "│ dense (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                        │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">11</span>)                  │             <span style=\"color: #00af00; text-decoration-color: #00af00\">132</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">6</span>)                   │              <span style=\"color: #00af00; text-decoration-color: #00af00\">72</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4</span>)                   │              <span style=\"color: #00af00; text-decoration-color: #00af00\">28</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense_3 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">1</span>)                   │               <span style=\"color: #00af00; text-decoration-color: #00af00\">5</span> │\n",
       "└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘\n",
       "</pre>\n"
      ],
      "text/plain": [
       "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓\n",
       "┃\u001b[1m \u001b[0m\u001b[1mLayer (type)                        \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1mOutput Shape               \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1m        Param #\u001b[0m\u001b[1m \u001b[0m┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩\n",
       "│ dense (\u001b[38;5;33mDense\u001b[0m)                        │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m11\u001b[0m)                  │             \u001b[38;5;34m132\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense_1 (\u001b[38;5;33mDense\u001b[0m)                      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m6\u001b[0m)                   │              \u001b[38;5;34m72\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense_2 (\u001b[38;5;33mDense\u001b[0m)                      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m4\u001b[0m)                   │              \u001b[38;5;34m28\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense_3 (\u001b[38;5;33mDense\u001b[0m)                      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m1\u001b[0m)                   │               \u001b[38;5;34m5\u001b[0m │\n",
       "└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Total params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">237</span> (948.00 B)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Total params: \u001b[0m\u001b[38;5;34m237\u001b[0m (948.00 B)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">237</span> (948.00 B)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Trainable params: \u001b[0m\u001b[38;5;34m237\u001b[0m (948.00 B)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Non-trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (0.00 B)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Non-trainable params: \u001b[0m\u001b[38;5;34m0\u001b[0m (0.00 B)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "e70bb9b4-3c25-40a3-9337-f2e1310d258f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.\n"
     ]
    }
   ],
   "source": [
    "model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "fab44628-622e-48a1-8bcd-ad7faa0a9b0e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(3500, 11)"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_X.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "17ba02bd-3c15-4773-a142-68ce32f7f270",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/20\n",
      "\u001b[1m350/350\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.8637 - loss: 0.3174\n",
      "Epoch 2/20\n",
      "\u001b[1m350/350\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 1ms/step - accuracy: 0.8629 - loss: 0.3151\n",
      "Epoch 3/20\n",
      "\u001b[1m350/350\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.8660 - loss: 0.3138\n",
      "Epoch 4/20\n",
      "\u001b[1m350/350\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.8683 - loss: 0.3123\n",
      "Epoch 5/20\n",
      "\u001b[1m350/350\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.8671 - loss: 0.3108\n",
      "Epoch 6/20\n",
      "\u001b[1m350/350\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.8643 - loss: 0.3094\n",
      "Epoch 7/20\n",
      "\u001b[1m350/350\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.8666 - loss: 0.3082\n",
      "Epoch 8/20\n",
      "\u001b[1m350/350\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.8654 - loss: 0.3076\n",
      "Epoch 9/20\n",
      "\u001b[1m350/350\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.8657 - loss: 0.3072\n",
      "Epoch 10/20\n",
      "\u001b[1m350/350\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.8680 - loss: 0.3062\n",
      "Epoch 11/20\n",
      "\u001b[1m350/350\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.8680 - loss: 0.3053\n",
      "Epoch 12/20\n",
      "\u001b[1m350/350\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.8634 - loss: 0.3055\n",
      "Epoch 13/20\n",
      "\u001b[1m350/350\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.8654 - loss: 0.3048\n",
      "Epoch 14/20\n",
      "\u001b[1m350/350\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.8660 - loss: 0.3039\n",
      "Epoch 15/20\n",
      "\u001b[1m350/350\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.8663 - loss: 0.3031\n",
      "Epoch 16/20\n",
      "\u001b[1m350/350\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.8666 - loss: 0.3029\n",
      "Epoch 17/20\n",
      "\u001b[1m350/350\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.8671 - loss: 0.3023\n",
      "Epoch 18/20\n",
      "\u001b[1m350/350\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.8629 - loss: 0.3021\n",
      "Epoch 19/20\n",
      "\u001b[1m350/350\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.8686 - loss: 0.3015\n",
      "Epoch 20/20\n",
      "\u001b[1m350/350\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.8660 - loss: 0.3013\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.src.callbacks.history.History at 0x182dc966fd0>"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(train_X, train_y, batch_size = 10, epochs = 20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "ee401ca6-08c6-4bb6-8ffc-aac21f6238ca",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m47/47\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 2ms/step\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([[0.12497811],\n",
       "       [0.1167827 ],\n",
       "       [0.33512014],\n",
       "       ...,\n",
       "       [0.6079354 ],\n",
       "       [0.51161164],\n",
       "       [0.0126922 ]], shape=(1500, 1), dtype=float32)"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "prob_test=model.predict(test_X)  # Probability \n",
    "prob_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "ee615a59-df1d-4ca5-ae1d-5f76ca07d7a1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.84      0.87      0.85       761\n",
      "           1       0.86      0.83      0.84       739\n",
      "\n",
      "    accuracy                           0.85      1500\n",
      "   macro avg       0.85      0.85      0.85      1500\n",
      "weighted avg       0.85      0.85      0.85      1500\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn import metrics\n",
    "y_test_pred=np.where(prob_test>0.5,1,0)\n",
    "print(metrics.classification_report(test_y, y_test_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bfdd8ef0-1c4a-4400-ad6d-32793c75c51a",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
