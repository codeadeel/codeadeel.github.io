import { Card, CardHeader, CardBody } from "@heroui/card";

/**
 * CardComponent
 * 
 * A reusable card component built using the HeroUI library. This component displays a card with a header and body, styled with Tailwind CSS classes.
 * 
 * Props:
 * - cardType (string): The type or category of the card, displayed in the header. Default is 'default'.
 * - cardTitle (string): The title or main content of the card, displayed in the body. Default is 'Default Title'.
 * - linkAddress (string): The URL the card links to. Default is '#'.
 * 
 * Example Usage:
 * <CardComponent cardType="Info" cardTitle="Learn More" linkAddress="/learn-more" />
 */

const CardComponent = ({ cardType = 'default', cardTitle = 'Default Title', linkAddress = '#' }) => {

  return (
    <a href={linkAddress} target="_parent" className="no-underline block transition-transform duration-300 transform hover:scale-105">
      <Card className="
      border
      border-gray-200
      rounded-2xl
      overflow-hidden
      w-full
      max-w-screen-sm
      h-auto
      bg-gradient-to-r from-gray-50 to-gray-100
      shadow-lg
      transform transition-transform duration-500 hover:rotate-1 hover:scale-90
      ">
        <CardHeader className="
        text-gray-700
        text-xs
        uppercase
        tracking-wide
        bg-gradient-to-r from-gray-200 to-gray-300
        p-2
        pl-3
        font-semibold
        rounded-t-2xl
        ">{cardType}</CardHeader>
        <CardBody className="
        text-black
        text-lg
        font-bold
        p-6
        break-words
        whitespace-normal
        overflow-wrap: break-word
        bg-white
        rounded-b-2xl
        ">{cardTitle}</CardBody>
      </Card>
    </a>
  );
};

export default CardComponent;
