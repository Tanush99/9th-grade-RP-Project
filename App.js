import { StatusBar } from 'expo-status-bar';
import React from 'react';
import { StyleSheet, Text, View, Image, br, h1, h2, p, a, Navigator, Button, TouchableOpacity } from 'react-native';
import {NavigationContainer} from '@react-navigation/native';
import {createStackNavigator} from '@react-navigation/stack';
import PredictScreen from './Predict';

const stack = createStackNavigator()

function HomeScreen({navigation}){
  return(
    <View>
      <Image style={{width: 1300, height: 1100}}
      source={require('./eye.jpg')}>
      </Image>
      <br></br>
      <br></br>
      <br></br>
      <br></br>
      <h1>Getting a Prediction of Whether or Not Retinitis Pigmentosa Is Present</h1>
      <br></br>
      <br></br>
      <br></br>
      <p>Use this app to get an accurate diagnosis of Retinitis Pigmentosa. Just take a retinal image using a smartphone attachment and upload it ,so the deep learning neural network can make a prediction. Then results will be displayed.</p>
      <br></br>
      <br></br>
      <Button title="Get Prediction" onPress={() => navigation.navigate('RP Predictor System')}/>
      <StatusBar style="auto" />
    </View>
  )
}

export default function App() {
  return (
    <NavigationContainer>
      <stack.Navigator initialRoutName="Home">
        <stack.Screen name="Home" component={HomeScreen}/>
        <stack.Screen name="RP Predictor System" component={PredictScreen}/>
      </stack.Navigator>
    </NavigationContainer>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});
