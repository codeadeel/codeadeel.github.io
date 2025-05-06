import { useState, useEffect } from 'react';
import axios from 'axios'; 
import CardComponent from './components/cardComponent';

/**
 * App Component
 * 
 * This component fetches repository data from a remote JSON file hosted on GitHub and displays it using the CardComponent.
 * It dynamically determines the data to display based on the current URL path.
 * 
 * Dependencies:
 * - useState: React hook for managing state.
 * - useEffect: React hook for side effects, such as fetching data.
 * - axios: HTTP client for making API requests.
 * - CardComponent: A child component used to render individual cards.
 * 
 * State:
 * - repoData: An array of objects representing the repository data to be displayed.
 * 
 * Side Effects:
 * - Fetches data from the GitHub JSON file when the component mounts.
 * - Updates the repoData state based on the current URL path.
 * 
 * Error Handling:
 * - Logs errors to the console if the data fetch fails.
 */

export default function App() {
  const [repoData, setRepoData] = useState([]);

  useEffect(() => {
    const currentUrl = window.parent.location.href.split("/")[3];
    axios.get("https://raw.githubusercontent.com/codeadeel/codeadeel.github.io/refs/heads/main/postsRepo.json").then((response) => {
      if(currentUrl===""){
        setRepoData(response.data["mainPage"]);
      } else {
        setRepoData(response.data[currentUrl]);
      }
    }
    ).catch((error) => {
      console.error("Error fetching data:", error);
    });
  }, []);

  return (
    <div className="flex flex-row flex-wrap justify-start gap-3 m-3">
      {repoData.map((data, index) => (
        <CardComponent
          key={index}
          cardType={data.category}
          cardTitle={data.title}
          linkAddress={data.url}
        />
      ))}
    </div>
  );
}
