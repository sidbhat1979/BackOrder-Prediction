import pandas
from file_operations import file_methods
from data_preprocessing import preprocessing
from data_ingestion import data_loader_prediction
from application_logging import logger
from Prediction_Raw_Data_Validation.predictionDataValidation import Prediction_Data_validation


class prediction:

    def __init__(self,path):
        self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        self.log_writer = logger.App_Logger()
        self.pred_data_val = Prediction_Data_validation(path)

    def predictionFromModel(self):

        try:
            self.pred_data_val.deletePredictionFile() #deletes the existing prediction file from last run!
            self.log_writer.log(self.file_object,'Start of Prediction')
            data_getter=data_loader_prediction.Data_Getter_Pred(self.file_object,self.log_writer)
            data=data_getter.get_data()

            preprocessor=preprocessing.Preprocessor(self.file_object,self.log_writer)
            data = preprocessor.remove_columns(data, ["Index_Product", "sku","oe_constraint"]) #removing oe_constraint as it was removed in training

            data = preprocessor.encodeCategoricalValuesPred(data)
            is_null_present=preprocessor.is_null_present(data)
            if(is_null_present):
                #data=preprocessor.impute_missing_values(data)
                data = data.dropna()

            data = preprocessor.scale_numerical_columns(data)

            data = preprocessor.pcaTransformation(data)
            file_loader=file_methods.File_Operation(self.file_object,self.log_writer)

            model_name = file_loader.find_correct_model_file()

            model = file_loader.load_model(model_name)
            result=list(model.predict(data))
            result = pandas.DataFrame(result, columns=['Prediction'])
            result["Prediction"] = result["Prediction"].map({ 0 : "Yes", 1: "No"})
            path="Prediction_Output_File/Predictions.csv"
            result.to_csv("Prediction_Output_File/Predictions.csv",header=True,mode='a+') #appends result to prediction file
            self.log_writer.log(self.file_object,'End of Prediction')
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the prediction!! Error:: %s' % ex)
            raise ex
        return path




