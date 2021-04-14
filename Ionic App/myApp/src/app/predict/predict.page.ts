import { Component, OnInit, ViewChild, ElementRef, Renderer2 } from '@angular/core';
import { Plugins, CameraResultType, CameraSource} from '@capacitor/core';
import { DomSanitizer, SafeResourceUrl} from '@angular/platform-browser';
import { Platform } from '@ionic/angular';
import { rendererTypeName } from '@angular/compiler';
import { defineCustomElements } from '@ionic/pwa-elements/loader';
import { Buffer } from 'buffer';
import * as tf from '@tensorflow/tfjs';

//const tr = require('@tensorflow/tfjs-node');

declare var require: any

const { Camera } = Plugins;
@Component({
  selector: 'app-home',
  templateUrl: 'predict.page.html',
  styleUrls: ['predict.page.scss'],
})
export class PredictPage {
   

  constructor(private sanitizer: DomSanitizer) {}
  
  ngOnInit(){
    this.loadModel();
  }

  async loadModel(){
    const model = await tf.loadLayersModel('../assets/model.json');
  }

  async takePicture(model) {
  
    const image = await Camera.getPhoto({
        quality: 90,
        allowEditing: true,
        resultType: CameraResultType.Uri,
        source: CameraSource.Camera});
      const photo = this.sanitizer.bypassSecurityTrustResourceUrl(image.base64String);
      const qwerty = this.sanitizer.bypassSecurityTrustResourceUrl(image.base64String);
      defineCustomElements(window);
      const b = Buffer.from(image.base64String, 'base64')
     
      
      
      // get the tensor(getting error from this line of code)
      let imr = tf.node.decodeImage(b)

      const output = model.predict(imr) as any;
   
      // Compute the output
      const prediction = Array.from(output.dataSync());
  
  }
  
}



