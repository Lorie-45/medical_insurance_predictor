import PredictorForm from "@/components/PredictorForm";

const Index = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-insurance-purple/5 to-insurance-orange/5">
      <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiM5MDYxZjkiIGZpbGwtb3BhY2l0eT0iMC4wMyI+PHBhdGggZD0iTTM2IDM0djZoNnYtNmgtNnptMCAwdi02aC02djZoNnptNiAwaDZ2LTZoLTZ2NnptLTEyIDBoLTZ2Nmg2di02em0tNi02aC02djZoNnYtNnoiLz48L2c+PC9nPjwvc3ZnPg==')] opacity-20"></div>
      
      <div className="container px-4 py-12 max-w-5xl mx-auto relative z-10">
        <div className="animate-fade-in text-center mb-12">
          <div className="inline-flex items-center justify-center bg-insurance-purple/10 rounded-full px-3 py-1 text-xs font-medium text-insurance-purple mb-4">
            Health Insurance Premium Calculator
          </div>
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-3 bg-insurance-gradient bg-clip-text text-transparent">
            Calculate Your Insurance Premium
          </h1>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Get an instant estimate of your health insurance premium based on your age, gender, and region.
            Our advanced prediction model provides accurate estimates tailored to your profile.
          </p>
        </div>

        <div className="animate-slide-up">
          <PredictorForm />
        </div>

        <div className="text-center text-xs text-muted-foreground mt-8 animate-fade-in">
          <p>This is a demonstration application using machine learning to predict insurance premiums.</p>
          <p>© 2023 Insurance Premium Calculator. All predictions are estimates only.</p>
        </div>
      </div>
    </div>
  );
};

export default Index;