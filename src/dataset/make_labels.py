import pandas as pd
import os

cwd = os.getcwd()
PATH = os.path.join(cwd, "../../data/full/labels/labels_csv_unprocessed.csv")
df = pd.read_csv(PATH)

df['cx'] = df["region_shape_attributes"].str.extract('"name":"point","cx":([^,"cy"]*)')
df['cy'] = df["region_shape_attributes"].str.extract('"cy":(\d+)')

df_piv = df.pivot(index="filename", columns="region_id", values=["cx", "cy"])
df_piv.to_csv(".temp/buffer.csv")
csv_final = pd.read_csv(".temp/buffer.csv")

csv_final = csv_final[
    ['cx', 'cy', 'cx.1', 'cy.1', 'cx.2', 'cy.2', 'cx.3', 'cy.3', 'cx.4', 'cy.4', 'cx.5', 'cy.5', 'cx.6', 'cy.6', 'cx.7',
     'cy.7', 'cx.8', 'cy.8', 'cx.9', 'cy.9', 'Unnamed: 0']]
csv_final.rename(
    columns={'cx': "0x", 'cy': '0y', 'cx.1': '1x', 'cy.1': '1y', 'cx.2': '2x', 'cy.2': '2y', 'cx.3': '3x', 'cy.3': '3y',
             'cx.4': '4x', 'cy.4': '4y', 'cx.5': '5x', 'cy.5': '5y', 'cx.6': '6x', 'cy.6': '6y', 'cx.7': '7x',
             'cy.7': '7y', 'cx.8': '8x', 'cy.8': '8y', 'cx.9': '9x', 'cy.9': '9y', 'Unnamed: 0': 'image'}, inplace=True)
csv_final.drop([0, 1], inplace=True)
csv_final.drop(["9x", "9y"], axis=1, inplace=True)

csv_final.to_csv(".temp/preprocessed_data.csv")
csv_final = pd.read_csv(".temp/preprocessed_data.csv")
csv_final.drop(['Unnamed: 0'], axis=1, inplace=True)

csv_final.to_csv(".temp/labels.csv", index=False)
print(csv_final)
