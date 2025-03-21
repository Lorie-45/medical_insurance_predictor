import { useState } from 'react';
import { Button } from '@components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Slider } from '@/components/ui/slider';
import { toast } from 'sonner';
import { predictInsurance } from '@/lib/mockPrediction';
import ResultDisplay from './ResultDisplay';

interface FormData {
  age: number;
  gender: string;
  region: string;
}

const PredictorForm = () => {
  const [formData, setFormData] = useState<FormData>({
    age: 30,
    gender: 'male',
    region: 'northeast'
  });
  const [prediction, setPrediction] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);

  const handleAgeChange = (values: number[]) => {
    setFormData(prev => ({ ...prev, age: values[0] }));
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleRadioChange = (name: string, value: string) => {
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.gender || !formData.region) {
      toast.error('Please fill out all fields');
      return;
    }

    setLoading(true);
    
    try {
      const result = await predictInsurance(formData);
      setPrediction(result);
      toast.success('Prediction completed successfully');
    } catch (error) {
      toast.error('Failed to get prediction');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-xl mx-auto">
      <form onSubmit={handleSubmit} className="glass-card p-6 space-y-6">
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <Label htmlFor="age" className="text-sm font-medium">
              Age: <span className="font-bold text-insurance-purple">{formData.age}</span>
            </Label>
            <Input
              type="number"
              name="age"
              value={formData.age}
              onChange={handleInputChange}
              min={18}
              max={100}
              className="w-20 input-glass text-right"
            />
          </div>
          <Slider
            id="age-slider"
            defaultValue={[30]}
            value={[formData.age]}
            min={18}
            max={100}
            step={1}
            onValueChange={handleAgeChange}
            className="py-4"
          />
        </div>

        <div className="space-y-3">
          <Label className="text-sm font-medium">Gender</Label>
          <RadioGroup
            defaultValue="male"
            value={formData.gender}
            onValueChange={(value) => handleRadioChange('gender', value)}
            className="grid grid-cols-2 gap-4"
          >
            <div className="radio-card flex items-center">
              <RadioGroupItem value="male" id="male" className="sr-only" />
              <Label
                htmlFor="male"
                className="flex items-center justify-center h-full w-full cursor-pointer"
              >
                <span className="font-medium">Male</span>
              </Label>
            </div>
            <div className="radio-card flex items-center">
              <RadioGroupItem value="female" id="female" className="sr-only" />
              <Label
                htmlFor="female"
                className="flex items-center justify-center h-full w-full cursor-pointer"
              >
                <span className="font-medium">Female</span>
              </Label>
            </div>
          </RadioGroup>
        </div>

        <div className="space-y-3">
          <Label className="text-sm font-medium">Region</Label>
          <RadioGroup
            defaultValue="northeast"
            value={formData.region}
            onValueChange={(value) => handleRadioChange('region', value)}
            className="grid grid-cols-2 gap-4"
          >
            <div className="radio-card flex items-center">
              <RadioGroupItem value="northeast" id="northeast" className="sr-only" />
              <Label
                htmlFor="northeast"
                className="flex items-center justify-center h-full w-full cursor-pointer"
              >
                <span className="font-medium">Northeast</span>
              </Label>
            </div>
            <div className="radio-card flex items-center">
              <RadioGroupItem value="northwest" id="northwest" className="sr-only" />
              <Label
                htmlFor="northwest"
                className="flex items-center justify-center h-full w-full cursor-pointer"
              >
                <span className="font-medium">Northwest</span>
              </Label>
            </div>
            <div className="radio-card flex items-center">
              <RadioGroupItem value="southeast" id="southeast" className="sr-only" />
              <Label
                htmlFor="southeast"
                className="flex items-center justify-center h-full w-full cursor-pointer"
              >
                <span className="font-medium">Southeast</span>
              </Label>
            </div>
            <div className="radio-card flex items-center">
              <RadioGroupItem value="southwest" id="southwest" className="sr-only" />
              <Label
                htmlFor="southwest"
                className="flex items-center justify-center h-full w-full cursor-pointer"
              >
                <span className="font-medium">Southwest</span>
              </Label>
            </div>
          </RadioGroup>
        </div>

        <div className="pt-2">
          <Button type="submit" disabled={loading} className="w-full btn-gradient">
            {loading ? 'Calculating...' : 'Calculate Premium'}
          </Button>
        </div>
      </form>

      <ResultDisplay amount={prediction} loading={loading} />
    </div>
  );
};

export default PredictorForm;