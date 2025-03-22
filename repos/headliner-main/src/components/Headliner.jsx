import React, { useState } from 'react';
import axios from 'axios';

const Headliner = () => {
  const [step, setStep] = useState('setup');
  const [preferences, setPreferences] = useState({
    topics: [],
    newsType: ''
  });
  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(false);

  // Available topics and news types
  const availableTopics = [
    'Politics', 'Business', 'Technology', 'Health', 
    'Science', 'Sports', 'Entertainment', 'Environment'
  ];
  
  const newsTypes = [
    'Breaking News', 'Top Headlines', 'In-depth Analysis', 'Opinion Pieces'
  ];

  // Handle topic selection
  const handleTopicToggle = (topic) => {
    setPreferences(prev => {
      if (prev.topics.includes(topic)) {
        return { ...prev, topics: prev.topics.filter(t => t !== topic) };
      } else {
        return { ...prev, topics: [...prev.topics, topic] };
      }
    });
  };

  // Handle news type selection
  const handleNewsTypeSelect = (type) => {
    setPreferences(prev => ({ ...prev, newsType: type }));
  };

  // Submit preferences and fetch news
  const handleSubmitPreferences = () => {
    if (preferences.topics.length === 0 || !preferences.newsType) {
      alert('Please select at least one topic and news type');
      return;
    }
    setStep('news');
    fetchNews();
  };

  // Simulated news fetching - in a real app, this would call an API
  const fetchNews = () => {
    setLoading(true);
    
    const API_KEY = 'b7cf5640c0fc4bc385c86fb11fba02b7'; // Replace with your actual API key
    
    const topicsQuery = preferences.topics.map(topic => `"${topic}"`).join(' OR ');
    
    // Map news type to appropriate API parameter
    let sortBy = 'publishedAt'; // default
    if (preferences.newsType === 'Top Headlines') {
      sortBy = 'popularity';
    } else if (preferences.newsType === 'In-depth Analysis') {
      sortBy = 'relevancy';
    }
    
    // Use the 'everything' endpoint to search by topics
    axios.get(`https://newsapi.org/v2/everything?q=${encodeURIComponent(topicsQuery)}&sortBy=${sortBy}&language=en&pageSize=20&apiKey=${API_KEY}`)
      .then(response => {
        const articles = response.data.articles.map((article, index) => ({
          id: index,
          title: article.title,
          description: article.description || 'No description available',
          source: article.source.name,
          topic: findMatchingTopic(article.title + ' ' + (article.description || '')),
          timestamp: formatTimestamp(article.publishedAt),
          url: article.url,
          publishedAt: article.publishedAt
        }));

        const sortedArticles = articles.sort((a, b) => {
          return new Date(b.publishedAt) - new Date(a.publishedAt);
        })
        
        setArticles(sortedArticles);
        setLoading(false);
      })
      .catch(error => {
        console.error('Error fetching news:', error);
        setLoading(false);
        // Handle errors gracefully in the UI
        alert('Failed to load news. Please try again later.');
      });
  };

  const findMatchingTopic = (content) => {
    content = content.toLowerCase();
    
    for (const topic of preferences.topics) {
      if (content.includes(topic.toLowerCase())) {
        return topic;
      }
    }
    
    // Default to first selected topic if no match found
    return preferences.topics[0];
  };
  
  // Helper function to format API timestamp to relative time
  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMinutes = Math.floor((now - date) / 60000);
    
    if (diffMinutes < 60) {
      return `${diffMinutes} minute${diffMinutes !== 1 ? 's' : ''} ago`;
    } else if (diffMinutes < 1440) {
      const hours = Math.floor(diffMinutes / 60);
      return `${hours} hour${hours !== 1 ? 's' : ''} ago`;
    } else {
      const days = Math.floor(diffMinutes / 1440);
      return `${days} day${days !== 1 ? 's' : ''} ago`;
    }
  };

  // Reset to preference selection
  const handleReset = () => {
    setStep('setup');
    setPreferences({
      topics: [],
      newsType: ''
    });
    setArticles([]);
  };

  // Get topic color class
  const getTopicColor = (topic) => {
    const colorMap = {
      'Politics': 'bg-red-100 text-red-800',
      'Business': 'bg-blue-100 text-blue-800',
      'Technology': 'bg-purple-100 text-purple-800',
      'Health': 'bg-green-100 text-green-800',
      'Science': 'bg-indigo-100 text-indigo-800',
      'Sports': 'bg-orange-100 text-orange-800',
      'Entertainment': 'bg-pink-100 text-pink-800',
      'Environment': 'bg-emerald-100 text-emerald-800'
    };
    
    return colorMap[topic] || 'bg-gray-100 text-gray-800';
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm">
        <div className="max-w-4xl mx-auto px-4 py-3 flex justify-center">
          {/* Replaced image with text-based logo */}
          <div className="font-sans font-black text-4xl tracking-tighter">
            headliner<span style={{ color: '#ff5042' }}>.</span>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-6">
        {step === 'setup' ? (
          <div className="space-y-8">
            <div className="text-center">
              <h2 className="font-sans text-2xl font-bold text-gray-900 mb-2">Personalize Your News Feed</h2>
              <p className="font-sans text-gray-600">Tell us what you're interested in and we'll curate the perfect news experience for you.</p>
            </div>
            
            <div className="bg-white p-6 rounded-xl shadow-md">
              <h3 className="font-sans text-xl font-semibold mb-4 text-gray-800">What topics interest you?</h3>
              <p className="font-sans text-gray-600 mb-4">Select all that apply. You can always change these later.</p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {availableTopics.map(topic => (
                  <button
                    key={topic}
                    onClick={() => handleTopicToggle(topic)}
                    className={`p-3 rounded-lg border-2 transition-all duration-200 ${
                      preferences.topics.includes(topic)
                        ? 'border-blue-500 bg-blue-50 text-blue-700 font-medium'
                        : 'border-gray-200 bg-white text-gray-700 hover:bg-gray-50'
                    }`}
                  >
                    {topic}
                  </button>
                ))}
              </div>
            </div>
            
            <div className="bg-white p-6 rounded-xl shadow-md">
              <h3 className="font-sans text-xl font-semibold mb-4 text-gray-800">News type preference</h3>
              <p className="font-sans text-gray-600 mb-4">Select the type of news content you prefer to see.</p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {newsTypes.map(type => (
                  <div 
                    key={type} 
                    onClick={() => handleNewsTypeSelect(type)}
                    className={`p-4 border-2 rounded-lg cursor-pointer ${
                      preferences.newsType === type
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:bg-gray-50'
                    }`}
                  >
                    <div className="flex items-center">
                      <div className={`w-5 h-5 rounded-full mr-2 flex items-center justify-center border ${
                        preferences.newsType === type ? 'border-blue-500 bg-blue-500' : 'border-gray-300'
                      }`}>
                        {preferences.newsType === type && (
                          <div className="w-2 h-2 bg-white rounded-full"/>
                        )}
                      </div>
                      <span className={preferences.newsType === type ? 'font-medium text-blue-700' : 'text-gray-700'}>
                        {type}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            <button
              onClick={handleSubmitPreferences}
              disabled={preferences.topics.length === 0 || !preferences.newsType}
              className={`w-full py-4 rounded-lg font-medium text-white text-lg transition-colors ${
                preferences.topics.length === 0 || !preferences.newsType
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-700'
              }`}
            >
              Show My Personalized News
            </button>
          </div>
        ) : (
          <div className="space-y-6">
            <div className="flex justify-between items-center">
              <div>
                <h2 className="font-sans text-2xl font-bold text-gray-900">Your News Feed</h2>
                <p className="font-sans text-gray-600">Personalized based on your preferences</p>
              </div>
              <button
                onClick={handleReset}
                className="px-4 py-2 bg-white border border-gray-300 rounded-lg text-gray-700 font-medium font-sans hover:bg-gray-50"
              >
                Edit Preferences
              </button>
            </div>
            
            <div className="bg-white p-4 rounded-xl shadow-sm">
              <div className="flex flex-wrap gap-2">
                {preferences.topics.map(topic => (
                  <span key={topic} className={`px-3 py-1 rounded-full font-sans text-sm font-medium ${getTopicColor(topic)}`}>
                    {topic}
                  </span>
                ))}
                <span className="px-3 py-1 rounded-full bg-gray-100 text-gray-800 text-sm font-medium font-sans">
                  {preferences.newsType}
                </span>
              </div>
            </div>
            
            {loading ? (
              <div className="flex flex-col items-center justify-center py-16 bg-white rounded-xl shadow-md">
                <div className="w-12 h-12 border-4 border-gray-200 border-t-blue-600 rounded-full animate-spin mb-4"></div>
                <p className="text-lg font-sans text-gray-700">Finding the latest headlines for you...</p>
              </div>
            ) : (
              <div className="space-y-4">
                {articles.length > 0 ? (
                  articles.map(article => (
                    <div key={article.id} className="bg-white p-5 rounded-xl shadow-md hover:shadow-lg transition-shadow">
                      <div className="flex items-center mb-3">
                        <span className={`px-2 py-1 rounded text-xs font-medium ${getTopicColor(article.topic)}`}>
                          {article.topic}
                        </span>
                        <span className="text-xs text-gray-500 ml-2">{article.timestamp}</span>
                      </div>
                      <h3 className="text-xl font-bold mb-2 text-gray-900">{article.title}</h3>
                      <p className="text-gray-700 mb-3">{article.description}</p>
                      <div className="flex justify-between items-center">
                        <span className="font-sans text-sm font-medium text-gray-500">Source: {article.source}</span>
                        <a 
                          href={article.url} 
                          target="_blank" 
                          rel="noopener noreferrer" 
                          className="text-blue-600 text-sm font-medium hover:text-blue-800"
                        >
                          Read more
                        </a>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-12 bg-white rounded-xl shadow-md">
                    <svg className="w-16 h-16 mx-auto text-gray-300 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                    </svg>
                    <p className="text-lg font-medium text-gray-700">No articles found matching your preferences</p>
                    <p className="text-gray-500 mt-1">Try selecting different topics or news types</p>
                    <button 
                      onClick={handleReset}
                      className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700"
                    >
                      Change Preferences
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </main>

      <footer className="bg-white py-4 mt-8 border-t">
      <div className="max-w-4xl mx-auto px-4 text-center text-gray-500 text-sm">
        Designed and developed by Sanika Kulkarni © {new Date().getFullYear()}
      </div>
      </footer>
    </div>
  );
};

export default Headliner;