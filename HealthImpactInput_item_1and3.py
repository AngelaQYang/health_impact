import pandas as pd
import numpy as np 
import h5py
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import glob, os
import pylab

model_dir = 'C:/Users/Angela/Documents/ITHIM/'
h5_base_file = 'daysim_outputs_2010.h5'
#h5_scen_file = 'daysim_outputs_2040.h5'
#h5_scen_file_pricing = 'daysim_outputs_2040_TransFu_15cents.h5'

def file_path(file_name):
    guide_file = model_dir + file_name
    return guide_file

daysim = h5py.File(file_path(h5_base_file), 'r')


############## Prepare data ################
#Convert H5 into DataFrame
def build_df(h5file, h5table, var_dict, survey_file=False):
    ''' Convert H5 into dataframe '''
    data = {}
    if survey_file:
        # survey h5 have nested data structure, different than daysim_outputs
        for col_name, var in var_dict.iteritems():
            data[col_name] = [i[0] for i in h5file[h5table][var][:]]
    else:
        for col_name, var in var_dict.iteritems():
            data[col_name] = [i for i in h5file[h5table][var][:]]
 
    return pd.DataFrame(data)


#where daysim= an h5 file you�ve read into memory already, and survey_file can be true or false, 
#depending on if the data is stored as an array or as scalar

tripdict={'Household ID': 'hhno',
            'Person Number': 'pno',
            'Travel Time':'travtime',
            'Travel Cost': 'travcost',
            'Travel Distance': 'travdist',
            'Mode': 'mode',
            'Purpose':'dpurp',
            'Departure Time': 'deptm',
            'Origin TAZ': 'otaz',
            'Destination TAZ': 'dtaz',
            'Value of Time': 'vot',
            'Departure Time': 'deptm',
            'Expansion Factor_tp': 'trexpfac'}
trip = build_df(h5file=daysim, h5table='Trip', var_dict=tripdict, survey_file=False)

persondict={'Household ID': 'hhno',
            'Person Number': 'pno',
            'Person Age':'pagey',
            'Person Gender': 'pgend'}
person = build_df(daysim,'Person', persondict, survey_file=False)

householddict={'Household ID': 'hhno',
            'Parcel': 'hhparcel',
            'Expansion Factor_hh': 'hhexpfac'}
household = build_df(h5file=daysim, h5table='Household', var_dict=householddict, survey_file=False)

print 'finished reading data from daysim file'

##########################person info##########################
#Create the unique id for identifing travellers
trip['person_id'] = trip['Household ID'].astype(str) + '_' + trip['Person Number'].astype(str) 
person['person_id'] = person['Household ID'].astype(str) + '_' + person['Person Number'].astype(str) 
print 'len(trip)', len(trip)
print 'unique person_id in trip', len(np.unique(person['person_id']))


#add age groups info to person
age_dict = {}
for age in np.unique(np.array(person['Person Age'])):
    if age > 0 and age <= 4:
        age_dict[age] = 0
    if age > 4 and age <= 14:
        age_dict[age] = 1
    if age > 14 and age <= 29:
        age_dict[age] = 2
    if age > 29 and age <= 44:
        age_dict[age] = 3
    if age > 44 and age <= 59:
        age_dict[age] = 4
    if age > 59 and age <= 69:
        age_dict[age] = 5
    if age > 69 and age <= 79:
        age_dict[age] = 6
    if age > 79:
        age_dict[age] = 7
person['age_group'] = person['Person Age'].map(age_dict)


################RCHL info##################################
#add RCHL info, filter the trips made by RCHL residents
right = household[['Household ID', 'Parcel']]
person_household = pd.merge(person, right, on='Household ID')
print len(person_household), len(person), len(right)

RCHL = pd.read_csv(r'H:\AY own records\4 health impact model\parcels_in_urbcens.csv', header=0, delimiter = ',', skiprows = 0)
RCHL.columns = ['Parcel', 'Name']
#delete the duplicated RCHL records
idx = np.unique(RCHL['Parcel'], return_index=True)[1]
right = RCHL.iloc[idx]
person_homeloc = pd.merge(person_household, right, how='left', on='Parcel')

print 'len: person_homeloc', len(person_homeloc)
print 'len: person_household (left)', len(person_household)
print 'len: RCHL (right)', len(RCHL)
print 'person_homeloc.head', person_homeloc.head()


#add person info to trips
right = person_homeloc[['person_id', 'Person Gender', 'Person Age', 'age_group', 'Parcel', 'Name']]
my_trip_data = pd.merge(trip, right, how='left', on='person_id')

print 'len: my_trip_data', len(my_trip_data)
print 'len: trip (left)', len(trip)
print 'len: person_homeloc (right)', len(person_homeloc)

print 'finsih prepareing trip-person-household tables'

####################### what you want to know #############
item = 'Travel Time'
#item = 'Travel Distance'
#item = 'Travel Cost'
print 'what you want to see:', item

###################### how you want to summerize ####################
##Bike/Walk trip length distribution (by time and distance)
bike = my_trip_data.loc[my_trip_data['Mode'] == 2]
walk = my_trip_data.loc[my_trip_data['Mode'] == 1]

print 'finished prepare bike and walk data tables'

def plot_as_you_like(axis_size, plot_path):
    fig, ax = plt.subplots()    
    x1 = bike[item][:2000]
    y1 = bike['Travel Distance'][:2000]
    x2 = walk[item][:2000]
    y2 = walk['Travel Distance'][:2000]
    ax.scatter(x1, y1, color = 'blue')
    ax.scatter(x2, y2, color = 'red')
    #figure settings
    plt.xlabel(item)
    plt.ylabel('Travel Distance')
    plt.axis(axis_size)
    plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    ax.legend(loc='lower right', shadow=True)
    #plt.figure(figsize=(10,5))
    pylab.rcParams['figure.figsize'] = (10.0, 10.0)

    pylab.savefig(plot_path)

#distributions for walk/bike trips within 60 mins & 20 miles
guide_file = file_path('2010_plot_bike_walk02182016')
plot_as_you_like([0, 60, 0, 20], guide_file)

print 'distribution figure done 2182016'

####### Average Time Spent Biking/Walking per Person
TOTPOP = len(person)
print 'TOTPOP:', TOTPOP

def average_cap(data):
    TOT = data[item].sum()
    per_cap = TOT / TOTPOP

    return per_cap
    
bike_per_cap = average_cap(bike)
walk_per_cap = average_cap(walk)
print 'bike_per_cap:', bike_per_cap
print 'walk_per_cap:', walk_per_cap

print 'average - per total capita done'

print 'finished calculate the whole average per capita data thing'

####### Average Time for person who spend time on Biking/Walking 
####### by Gender/Age Group/Regional Center Home location

def average_person(data, by_type):
    data_table = data.pivot_table(item, columns = by_type, aggfunc=[np.sum])
    data_table[by_type] = data_table.index
    print 'data_table index set'
    dict_RCHLparcel = {}
    for ele in np.unique(data_table[by_type]):
        people_c = data.loc[data[by_type] == ele]
        dict_RCHLparcel[ele] = len(np.unique(np.array(people_c['person_id'])))
        #print ele 
    data_table['person_count'] = data_table[by_type].map(dict_RCHLparcel) 
    dict_RCHLname = {}
    if by_type == 'Parcel':
        dict_RCHLname = data.set_index('Parcel')['Name'].to_dict()
        data_table['Name'] = data_table['Parcel'].map(dict_RCHLname)
    print 'person_count set'
    data_table['averg'] = data_table['sum'] / data_table['person_count']

    return data_table

bike_gender_averg = average_person(bike, 'Person Gender')
walk_gender_averg = average_person(walk, 'Person Gender')
bike_age_averg = average_person(bike, 'age_group')
walk_age_averg = average_person(walk, 'age_group')
bike_RCHL = bike.dropna()
walk_RCHL = walk.dropna()
bike_RCHL_averg = average_person(bike_RCHL, 'Parcel')
walk_RCHL_averg = average_person(walk_RCHL, 'Parcel')

print 'average per person done'

######## per capita biking/walking
#total person size grouped by gender/age/RCHL

TOTPOP_gender = person.groupby('Person Gender').size()
TOTPOP_age = person.groupby('age_group').size()
person_RCHL = person_homeloc.dropna()
TOTPOP_RCHL = person_RCHL.groupby('Parcel').size()

print 'TOTPOP_gender', TOTPOP_gender
print 'TOTPOP_age', TOTPOP_age
print 'len: TOTPOP_RCHL', len(TOTPOP_RCHL)

def join_info(left, right, by_type):
    df_right = pd.DataFrame(right, columns = ['TOTPOP'])
    df_right[by_type] = df_right.index
    result = pd.merge(left, df_right, on = by_type)
    capname = by_type + '_cap'
    result[capname] = result['sum']/result['TOTPOP']
    print capname, 'value set'

    return result

bike_gender_person = join_info(bike_gender_averg, TOTPOP_gender, 'Person Gender')
walk_gender_person = join_info(walk_gender_averg, TOTPOP_gender, 'Person Gender')
bike_age_person = join_info(bike_age_averg, TOTPOP_age, 'age_group')
walk_age_person = join_info(walk_age_averg, TOTPOP_age, 'age_group')
bike_RCHL_person = join_info(bike_RCHL_averg, TOTPOP_RCHL, 'Parcel')
walk_RCHL_person = join_info(walk_RCHL_averg, TOTPOP_RCHL, 'Parcel')

print 'average person by groups done'

################# Save summerize tables ###################
bike_gender_person.to_csv(file_path('2010_time_bike_gender0218.csv'))
walk_gender_person.to_csv(file_path('2010_time_walk_gender0218.csv'))
bike_age_person.to_csv(file_path('2010_time_bike_age0218.csv'))
walk_age_person.to_csv(file_path('2010_time_walk_age0218.csv'))
bike_RCHL_person.to_csv(file_path('2010_time_bike_RCHL0218.csv'))
walk_RCHL_person.to_csv(file_path('2010_time_walk_RCHL0218.csv'))

print 'saved the tables, and this is the end'

########### write the data out to csv ###########
#def save_to_csv(data, col_name, guide_file):
#    result = pd.DataFrame(data[col_name], columns = [col_name])
#    return result.to_csv(guide_file)


#cap_df_name_dict = {'bike_time_gender_cap': bike_time_gender_cap, 
#    'walk_time_gender_cap': walk_time_gender_cap, 
#    'bike_time_age_cap': bike_time_age_cap, 
#    'walk_time_age_cap': walk_time_age_cap, 
#    'bike_time_RCHL_cap': bike_time_RCHL_cap, 
#    'walk_time_RCHL_cap':  walk_time_RCHL_cap}

#for df_name in cap_df_name_dict: 
#    guide_file = file_path(df_name+'.csv')
#    save_to_csv(cap_df_name_dict[df_name], cap_df_name_dict[df_name].columns[-4:], guide_file)
#    print df_name, 'done'
#    #'mean' -- per person
#    #'cap' -- per capita


#print 'done'




























#df_bikeperson = pd.DataFrame({'Person ID':bike['Person ID'], 'Person Gender':bike['Person Gender'], 'Person Age':bike['Person Age'], 'Person Number':bike['Person Number_y']})
##df_person = np.unique(df_person[['Person ID', 'Person Number']])
#print len(df_bikeperson)
#df_bikeperson.columns 
##df_person_info = df_person.set_index('Person ID')

#df1 = pd.DataFrame({'Person ID': bike['Person ID'], 'Travel Time': bike['Travel Time'], 'Travel Distance': bike['Travel Distance']})
#df_biketrip = df1.groupby(['Person ID']).sum()
#df_biketrip['Person ID'] = df_biketrip.index
#print len(df_biketrip)
#df_biketrip.columns

#result = pd.merge(df_biketrip, df_bikeperson, how = 'left', on = 'Person ID', lsuffix="_review" )
#len(result)

