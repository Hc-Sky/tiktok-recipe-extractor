import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import App from './App';

// Mock fetch
global.fetch = jest.fn();

describe('App', () => {
  beforeEach(() => {
    fetch.mockClear();
  });

  it('renders the app title', () => {
    render(<App />);
    expect(screen.getByText('TikTok Recipe Extractor')).toBeInTheDocument();
  });

  it('shows error when submitting without URL', async () => {
    render(<App />);

    // Click the submit button without entering a URL
    fireEvent.click(screen.getByText('Extract Recipe'));

    // Check if error message is displayed
    expect(await screen.findByText('Please enter a TikTok URL')).toBeInTheDocument();
  });

  it('submits form with URL and profile', async () => {
    // Mock successful response
    const mockRecipe = {
      titre: 'Test Recipe',
      résumé: 'A test recipe',
      ingrédients: [{ nom: 'Test Ingredient', quantité: '100g' }],
      étapes: ['Step 1', 'Step 2'],
      temps: { préparation: '10 min', cuisson: '20 min' },
      nutrition: { calories: 500, protéines: 20, lipides: 10, glucides: 30 },
      évaluation_santé: 'Good for test',
      pdf_link: '/download/test.pdf'
    };

    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockRecipe
    });

    render(<App />);

    // Enter URL
    fireEvent.change(screen.getByPlaceholderText('https://www.tiktok.com/@username/video/1234567890'), {
      target: { value: 'https://www.tiktok.com/@test/video/123456' }
    });

    // Select profile
    fireEvent.click(screen.getByText('Select a profile'));
    fireEvent.click(screen.getByText('Muscle Gain'));

    // Submit form
    fireEvent.click(screen.getByText('Extract Recipe'));

    // Wait for fetch to be called
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(expect.stringContaining('/extract'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          url: 'https://www.tiktok.com/@test/video/123456',
          profil_utilisateur: 'prise de masse'
        }),
      });
    });

    // Check if recipe is displayed
    expect(await screen.findByText('Test Recipe')).toBeInTheDocument();
  });

  it('handles API errors', async () => {
    // Mock error response
    fetch.mockResolvedValueOnce({
      ok: false
    });

    render(<App />);

    // Enter URL
    fireEvent.change(screen.getByPlaceholderText('https://www.tiktok.com/@username/video/1234567890'), {
      target: { value: 'https://www.tiktok.com/@test/video/123456' }
    });

    // Submit form
    fireEvent.click(screen.getByText('Extract Recipe'));

    // Check if error message is displayed
    expect(await screen.findByText('Failed to extract recipe')).toBeInTheDocument();
  });
});
