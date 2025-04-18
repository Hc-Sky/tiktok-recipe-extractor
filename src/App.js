import React, { useState } from 'react';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Label } from './components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui/select';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from './components/ui/card';

function App() {
  const [link, setLink] = useState('');
  const [profile, setProfile] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const API_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000';

  const handleSubmit = async () => {
    if (!link) {
      setError('Please enter a TikTok URL');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await fetch(`${API_URL}/extract`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          url: link,
          profil_utilisateur: profile || undefined
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to extract recipe');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background p-4 md:p-8">
      <div className="container mx-auto max-w-4xl">
        <header className="mb-8 text-center">
          <h1 className="text-3xl font-bold tracking-tight mb-2">TikTok Recipe Extractor</h1>
          <p className="text-muted-foreground">Generate recipe cards from TikTok videos</p>
        </header>

        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Extract Recipe</CardTitle>
            <CardDescription>
              Enter a TikTok URL and select your dietary profile to get a personalized recipe card
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="tiktok-url">TikTok URL</Label>
              <Input
                id="tiktok-url"
                type="text"
                value={link}
                onChange={(e) => setLink(e.target.value)}
                placeholder="https://www.tiktok.com/@username/video/1234567890"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="profile">Dietary Profile (Optional)</Label>
              <Select value={profile} onValueChange={setProfile}>
                <SelectTrigger id="profile">
                  <SelectValue placeholder="Select a profile" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="perte de poids">Weight Loss</SelectItem>
                  <SelectItem value="prise de masse">Muscle Gain</SelectItem>
                  <SelectItem value="végétarien">Vegetarian</SelectItem>
                </SelectContent>
              </Select>
            </div>
            {error && <p className="text-destructive text-sm">{error}</p>}
          </CardContent>
          <CardFooter>
            <Button onClick={handleSubmit} disabled={loading}>
              {loading ? 'Extracting...' : 'Extract Recipe'}
            </Button>
          </CardFooter>
        </Card>

        {result && (
          <Card>
            <CardHeader>
              <CardTitle>{result.titre}</CardTitle>
              <CardDescription>{result.résumé}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <h3 className="font-medium text-lg mb-2">Ingredients</h3>
                <ul className="list-disc pl-5 space-y-1">
                  {result.ingrédients.map((ingredient, index) => (
                    <li key={index}>
                      {ingredient.nom}: {ingredient.quantité}
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <h3 className="font-medium text-lg mb-2">Steps</h3>
                <ol className="list-decimal pl-5 space-y-1">
                  {result.étapes.map((step, index) => (
                    <li key={index}>{step}</li>
                  ))}
                </ol>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h3 className="font-medium text-lg mb-2">Time</h3>
                  <p>Preparation: {result.temps.préparation}</p>
                  <p>Cooking: {result.temps.cuisson}</p>
                </div>

                <div>
                  <h3 className="font-medium text-lg mb-2">Nutrition</h3>
                  <p>Calories: {result.nutrition.calories} kcal</p>
                  <p>Protein: {result.nutrition.protéines}g</p>
                  <p>Fat: {result.nutrition.lipides}g</p>
                  <p>Carbs: {result.nutrition.glucides}g</p>
                </div>
              </div>

              <div>
                <h3 className="font-medium text-lg mb-2">Health Evaluation</h3>
                <p>{result.évaluation_santé}</p>
              </div>
            </CardContent>
            <CardFooter>
              <Button asChild variant="outline">
                <a href={`${API_URL}${result.pdf_link}`} download="recipe.pdf">
                  Download PDF
                </a>
              </Button>
            </CardFooter>
          </Card>
        )}
      </div>
    </div>
  );
}

export default App;
