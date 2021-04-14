import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { IonicModule } from '@ionic/angular';
import { PredictPageRoutingModule } from './predict-routing.module';
import { Component, OnInit, ViewChild, ElementRef, Renderer2 } from '@angular/core';
import { Plugins, CameraResultType, CameraSource} from '@capacitor/core';
import { DomSanitizer, SafeResourceUrl} from '@angular/platform-browser';
import { Platform } from '@ionic/angular';
import * as tf from '@tensorflow/tfjs';
import { tensor1d } from '@tensorflow/tfjs';
import { rendererTypeName } from '@angular/compiler';
import { defineCustomElements } from '@ionic/pwa-elements/loader';

const { Camera } = Plugins;

@Component({
  selector: 'app-predict',
  templateUrl: './predict.page.html',
  styleUrls: ['./predict.page.scss'],
})

@NgModule({
  imports: [
    CommonModule,
    FormsModule,
    IonicModule,
    PredictPageRoutingModule
  ],
  
})
export class PredictPageModule {

  
}
