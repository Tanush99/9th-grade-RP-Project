import { StatusBar } from 'expo-status-bar';
import React from 'react';
import { StyleSheet, Text, View, Image, TouchableOpacity, Alert } from 'react-native';
import {createStackNavigator} from '@react-navigation/stack';
import {Camera} from 'expo-camera'
import {CAMERA_ROLL} from 'expo-camera'
import * as ImagePicker from 'expo-image-picker';
import Constants from 'expo-constants';
import * as Permissions from 'expo-permissions';
//import { fetch, decodeJpeg } from '@tensorflow/tfjs-react-native';

const stack = createStackNavigator()
  
  const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: '#fff',
      alignItems: 'center',
      justifyContent: 'center',
    },
  });


const PickFromCamera = async () => {
    const granted = await Permissions.askAsync(Permissions.CAMERA)
    if (granted){
        let data = await ImagePicker.launchCameraAsync({
            mediaTypes:ImagePicker.MediaTypeOptions.Images,
            allowsEditing: true,
            aspect: [1,1],
            quality:0.5
        })
        console.log(data)
        // const model = await tf.loadLayersModel('./model.json');
        // const image = data;
        // const imageAssetPath = Image.resolveAssetSource(image);
        // const response = await fetch(imageAssetPath.uri, {}, { isBinary: true });
        // const imageData = await response.arrayBuffer();
        // const imageTensor = decodeJpeg(imageData);
        // const prediction = await model.classify(imageTensor);
    }else{
        Alert.alert("Permissions must be given to the predictor system")
    }
    }


    export default function PredictScreen() {
        return (
         
          <View style={styles.container}>
              <View style={{
              flex: 1,
              backgroundColor: '#fff',
              justifyContent: 'center',
              alignItems: 'center'
                }}>
                <StatusBar style="auto" />
                <TouchableOpacity
                onPress={PickFromCamera}
                style={{
                width: 130,
                borderRadius: 4,
                backgroundColor: '#14274e',
                flexDirection: 'row',
                justifyContent: 'center',
                alignItems: 'center',
                height: 40
              }}
            >
              <Text
                style={{
                  color: '#fff',
                  fontWeight: 'bold',
                  textAlign: 'center'
                }}
              >
                Take picture
              </Text>
            </TouchableOpacity>
                <Text></Text>
              </View>
          </View>
        );
      }






