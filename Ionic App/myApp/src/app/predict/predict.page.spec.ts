import { async, ComponentFixture, TestBed } from '@angular/core/testing';
import { IonicModule } from '@ionic/angular';

import { PredictPage } from './predict.page';

describe('PredictPage', () => {
  let component: PredictPage;
  let fixture: ComponentFixture<PredictPage>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ PredictPage ],
      imports: [IonicModule.forRoot()]
    }).compileComponents();

    fixture = TestBed.createComponent(PredictPage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  }));

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
