import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import os

# ------------------------------------ General -------------------------------
def progress_bar(current, total, bar_length=20):
    """
    Helper function for debugging.
    Shows update
    """
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '

    ending = '\n' if current == total else '\r'

    print(f'Progress: [{arrow}{padding}] {int(fraction*100)}%', end=ending)


# ------------------------------------ DB Creation -------------------------------
def newtimes(db: pd.DataFrame) -> pd.DataFrame:
    """
    Standartize the counts per time values by summing values per standartized time interval
    Inputs: DF: Time, Count
    Outputs: DF : Time, Count
    """
    delta = 0.0065
    times = np.arange(1, 20, delta)
    counts = []

    for time in times:
        counts.append(sum(db.loc[np.logical_and(db["Time"] >= time, db["Time"] < time + delta), "Count"]))
    fixed_db = pd.DataFrame(data=np.array([times, counts])).T
    fixed_db.columns = db.columns
    return fixed_db


def db_manyfiles(name: str, data: dict, offset: int):
    """
    Opens known spectra, replaces time domains with standardized times and saves List[DF] as a json
    with the following name: "sorted_",name,"spectra.json"
    :param name: name of arson type
    :param path: path that contains known spectra

    :return: array of spectra, a list of targets
    """

    DBpath = data[1]

    files = os.listdir(DBpath)
    all_dbs = []
    for ix, file in enumerate(files):
        if 'csv' in file:
            db = pd.read_csv(DBpath+"\\"+file, skiprows=1, header=0).reset_index()
            db.columns = ["Time", "Count"]
            db = newtimes(db)
            db["Class"] = name
            db["SampleID"] = ix+offset
            all_dbs.append(db)
    return pd.concat(all_dbs)


def db_onefiles(name: str, data: dict, offset: int) -> pd.DataFrame:
    """
    Opens known spectra, replaces time domains with standardized times and returns a DF
    :param name: name of arson type
    :param path: path that contains excel file
    :return: DF.
    """

    DBpath = data[1]
    xl = pd.ExcelFile(DBpath)
    all_dbs = []
    for ix, sheet in enumerate(xl.sheet_names):
        db = xl.parse(sheet, skiprows=2, header=None)
        db.columns = ["Time", "Count"]
        db = newtimes(db)
        db["Class"] = name
        db["SampleID"] = ix+offset
        all_dbs.append(db)
    return pd.concat(all_dbs)


def load_dbs(arson_types) -> dict:
    """
    Runs on all of the arson types and creates a list of DF containing the spectra, in standartized, time units
    :param arson_types: contains the type an the path of spectra
    :return: a list of DF containing the spectra
    """
    all_dbs = []
    offset = 0
    for name, data in arson_types.items():
        if data[0] == "one":
            all_dbs.append(db_onefiles(name, data, offset))
        elif data[0] == "many":
            all_dbs.append(db_manyfiles(name, data, offset))

        offset = max(all_dbs[-1]["SampleID"]) + 1
    return(pd.concat(all_dbs))


# ------------------------------------ Preprocess -------------------------------

def preprocess_data(db: pd.DataFrame, SIZE_OF_SPECTRA : int, labler = None) -> (np.ndarray, np.ndarray):
    """
    :param db: The database that needs to be preprocessed. Contains "Time", "Count", "Class", SampleID
    :param SIZE_OF_SPECTRA: constant number of samples per spectrum
    :param labler: a categorical labler
    :return:
    realSpectra: The sored db with the format of A[i,j] where i is the sampleID and j is the value at timestamp j
     class_vec: A vector of the encoded classes
    label_encoder: a categorical labler
    """
    realSpectra = np.array(db["Count"]).reshape(-1, SIZE_OF_SPECTRA)

    # This can be done better if DB is rearranged differently
    class_vec = []
    if labler:
        label_encoder = labler
    else:
        label_encoder = LabelEncoder()
    sampleIDs = list(set(db["SampleID"]))
    sampleIDs.sort()
    for sampleNum in sampleIDs:
        class_vec.append(db[db["SampleID"] == sampleNum]["Class"].iloc[0])  # Take the registry of the first class
    class_vec = label_encoder.fit_transform(np.array(class_vec))
    return realSpectra, class_vec, label_encoder


def logNorm(spectra, realSpectraBounds):
    output = np.log(spectra)
    output[np.isinf(output)] = 0
    output = output/(realSpectraBounds[1]-realSpectraBounds[0])+realSpectraBounds[0]
    output[output > 1] = 1
    output[output < 0] = 0
    return output

# ------------------------------------ Synthetic Generation -------------------------------


def generate_spectrum(realSpectra: np.ndarray, classification, batch_size: int) -> np.ndarray:
    """
    :param realSpectra:
    :return:
    """

    spectra = []
    gen_class = []
    for i in range(batch_size):

        progress_bar(i, batch_size)
        chosenType = np.random.randint(0, len(set(classification)))  # len(set(classification)) is the number of classes
        # Subset the data to only include spectra from a specific type
        subsetSpectra = realSpectra[classification == chosenType, :]
        numSamples = subsetSpectra.shape[0]
        # We draw randomly draw sampleID's. The number of the sampleIDs to draw is also randomly chosen.
        numOfSpectraToDraw = np.random.randint(0, numSamples)
        chosenSpectra = np.random.randint(0, numSamples, numOfSpectraToDraw)
        weights = np.random.randint(-4, 4, numOfSpectraToDraw).reshape(-1, 1) * np.random.random()
        # (2924 X weights) dot (weights X 1) = 2924 X 1
        spectra.append(np.abs(np.dot(subsetSpectra[chosenSpectra, :].T, weights)))
        gen_class.append(chosenType)
    return np.array(spectra), np.array(gen_class)


# ------------------------------------ DL Generation -------------------------------
def makeModelDyn(input_shape, L = 3):
    model_t = keras.Sequential()
    if L <1:
        raise("L can not be less than 1")
    model_t.add(keras.Input(shape=input_shape))
    for i in range(L):
        model_t.add(keras.layers.Dense(int(input_shape*(0.8**i)), activation='relu'))
    model_t.add(keras.layers.Dense(3, activation='softmax'))
    return model_t


if __name__ == "__main__":

    batch1 = {'BZ': ['many', 'Data/BZ - TRAIN'],
                   'PD': ['many', 'Data/pd - Train'],
                   'HR': ['one', 'Data/HOLER - TRAIN/HOLER-train.xlsx']
    }

    batch2 = {'BZ': ['one', 'Data/BZ - TEST/BZ.xlsx'],
                   'PD': ['one', 'Data/pd - Test/pd.xlsx'],
                   'HR': ['one', 'Data/HOLER - TEST/HOLER1.xlsx']
    }

    test = {'BZ': ['one', 'Data/BZ - TEST/BZ.xlsx'],
                   'PD': ['one', 'Data/pd - Test/pd.xlsx'],
                   'HR': ['one', 'Data/HOLER - TEST/HOLER1.xlsx']
    }

    try:
        pd.read_csv("Outputs/db.csv")
    except:
        print("Training file not found, making a new one")
        dbs_batch1 = load_dbs(batch1)
        dbs_batch2 = load_dbs(batch2)
        dbs_batch2["SampleID"] += max(dbs_batch1["SampleID"])+1
        db = pd.concat([dbs_batch1, dbs_batch2])
        db.to_csv("Outputs/db.csv", index = False)

    try:
        pd.read_csv("Outputs/test.csv")
    except:
        print("Testing file not found, making a new one")
        dbs_test = load_dbs(batch1)
        dbs_test.to_csv("Outputs/test.csv", index = False)
